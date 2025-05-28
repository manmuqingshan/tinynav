import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path, Odometry
import numpy as np
import sys
from math_utils import matrix_to_quat, msg2np, np2msg
from sensor_msgs.msg import Image, CameraInfo
from message_filters import TimeSynchronizer, Subscriber
from cv_bridge import CvBridge
import cv2
from codetiming import Timer
import os
import argparse

from tinynav.tinynav_cpp_bind import pose_graph_solve
from models_trt import LightGlueTRT, Dinov2TRT, SuperPointTRT
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
from std_msgs.msg import Header
from planning_node import run_raycasting_loopy
import logging

logger = logging.getLogger(__name__)

def project_3d_to_2d(points_3d: np.ndarray, T_camera_world: np.ndarray, K: np.ndarray):
    '''
    points_3d: (N, 3)
    T_camera_world: (4, 4)
    K: (3, 3)
    return: (N, 2)
    '''
    rotation_matrix = T_camera_world[:3, :3]
    translation = T_camera_world[:3, 3]
    points_3d_in_camera = (rotation_matrix @ points_3d.T).T + translation

    points_2d = points_3d_in_camera @ K.T
    points_2d = points_2d[:, :2] / points_2d[:, 2:3]
    return points_2d

def project_3d_to_2d_into_image(image, point_2ds, point_3ds, T_camera_world, K):
    projected_2ds = project_3d_to_2d(point_3ds, T_camera_world, K)
    for point_2d, projected_2d in zip(point_2ds, projected_2ds):
        cv2.circle(image, (int(projected_2d[0]), int(projected_2d[1])), 5, (0, 255, 0), -1)
        cv2.circle(image, (int(point_2d[0]), int(point_2d[1])), 3, (0, 0, 255), -1)
    return image

def draw_image_match_origin(prev_image: np.ndarray, curr_image: np.ndarray, prev_keypoints: np.ndarray, curr_keypoints: np.ndarray, matches: np.ndarray):
    cv_matches = [cv2.DMatch(_queryIdx=matches[index, 0].item(), _trainIdx=matches[index, 1].item(), _imgIdx=0, _distance=0) for index in range(matches.shape[0])]
    # convert kpts_prev and kpts_curr to cv2.KeyPoint 
    cv_kpts_prev = [cv2.KeyPoint(x=prev_keypoints[index, 0].item(), y=prev_keypoints[index, 1].item(), size=20) for index in range(prev_keypoints.shape[0])]
    cv_kpts_curr = [cv2.KeyPoint(x=curr_keypoints[index, 0].item(), y=curr_keypoints[index, 1].item(), size=20) for index in range(curr_keypoints.shape[0])]
    output_image = cv2.drawMatches(prev_image, cv_kpts_prev, curr_image, cv_kpts_curr, cv_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    return output_image

def depth_to_cloud(depth: np.ndarray, K: np.ndarray) -> np.ndarray:
    """
    Convert depth image to point cloud.
    :param depth: (H, W) depth image.
    :param K: (3, 3) camera intrinsic matrix.
    :return: (N, 3) point cloud in camera coordinates.
    """
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    z = depth.flatten()
    
    x = (u.flatten() - K[0, 2]) * z / K[0, 0]
    y = (v.flatten() - K[1, 2]) * z / K[1, 1]
    
    points_3d = np.vstack((x, y, z)).T
    return points_3d[~np.isnan(points_3d).any(axis=1)]

def transform_point_cloud(point_cloud: np.ndarray, T: np.ndarray) -> np.ndarray:
    """
    Transform a point cloud with a transformation matrix.
    :param point_cloud: (N, 3) numpy array of points in the point cloud.
    :param T: (4, 4) transformation matrix.
    :return: (N, 3) transformed point cloud.
    """
    assert point_cloud.shape[1] == 3, "Point cloud must be of shape (N, 3)"
    assert T.shape == (4, 4), "Transformation matrix must be of shape (4, 4)"
    
    # Convert to homogeneous coordinates
    ones = np.ones((point_cloud.shape[0], 1))
    homogeneous_points = np.hstack((point_cloud, ones))
    # Apply transformation
    transformed_points = homogeneous_points @ T.T
    return transformed_points[:, :3]

def merge_grids(grid1:np.ndarray, grid1_origin:np.ndarray, grid2:np.ndarray, grid2_origin:np.ndarray, resolution:float) -> tuple[np.ndarray, np.ndarray]:
        """
        Merge two grids into one.
        """
        min_x = min(grid1_origin[0], grid2_origin[0])
        min_y = min(grid1_origin[1], grid2_origin[1])
        min_z = min(grid1_origin[2], grid2_origin[2])

        max_x = max(grid1_origin[0] + grid1.shape[0] * resolution, grid2_origin[0] + grid2.shape[0] * resolution)
        max_y = max(grid1_origin[1] + grid1.shape[1] * resolution, grid2_origin[1] + grid2.shape[1] * resolution)
        max_z = max(grid1_origin[2] + grid1.shape[2] * resolution, grid2_origin[2] + grid2.shape[2] * resolution)

        new_shape_x = int((max_x - min_x) / resolution) + 1
        new_shape_y = int((max_y - min_y) / resolution) + 1
        new_shape_z = int((max_z - min_z) / resolution) + 1

        new_grid = np.zeros((new_shape_x, new_shape_y, new_shape_z), dtype=np.float32)
        new_origin = np.array([min_x, min_y, min_z], dtype=np.float32)
        grid1_x_start = int((grid1_origin[0] - min_x) / resolution)
        grid1_y_start = int((grid1_origin[1] - min_y) / resolution)
        grid1_z_start = int((grid1_origin[2] - min_z) / resolution)
        grid2_x_start = int((grid2_origin[0] - min_x) / resolution)
        grid2_y_start = int((grid2_origin[1] - min_y) / resolution)
        grid2_z_start = int((grid2_origin[2] - min_z) / resolution)
        # Merge the grids
        new_grid[grid1_x_start:grid1_x_start + grid1.shape[0],
                  grid1_y_start:grid1_y_start + grid1.shape[1],
                  grid1_z_start:grid1_z_start + grid1.shape[2]] += grid1

        new_grid[grid2_x_start:grid2_x_start + grid2.shape[0],
                  grid2_y_start:grid2_y_start + grid2.shape[1],
                  grid2_z_start:grid2_z_start + grid2.shape[2]] += grid2
        return new_grid, new_origin

class MapNode(Node):
    def __init__(self, mapping_mode):
        super().__init__('map_node')
        self.super_point_extractor = SuperPointTRT()
        self.light_glue_matcher = LightGlueTRT()
        self.dinov2_model = Dinov2TRT()

        self.bridge = CvBridge()

        self.disp_sub = Subscriber(self, Image, '/slam/keyframe_disparity')
        self.keyframe_image_sub = Subscriber(self, Image, '/slam/keyframe_image')
        self.keyframe_odom_sub = Subscriber(self, Odometry, '/slam/keyframe_odom')

        self.pose_graph_trajectory_pub = self.create_publisher(Path, "/mapping/pose_graph_trajectory", 10)

        self.relocation_pub = self.create_publisher(Odometry, '/map/relocalization', 10)

        self.project_3d_to_2d_pub = self.create_publisher(Image, "/mapping/project_3d_to_2d", 10)
        self.matches_image_pub = self.create_publisher(Image, "/mapping/keyframe_matches_images", 10)
        self.loop_matches_image_pub = self.create_publisher(Image, "/mapping/loop_matches_images", 10)

        self.global_map_pub = self.create_publisher(PointCloud2, "/mapping/global_map", 10)
        self.global_map_timer = self.create_timer(10.0, self.global_map_callback)
        self.point_cloud = None

        self.ts = TimeSynchronizer([self.keyframe_image_sub, self.keyframe_odom_sub, self.disp_sub], 1000)
        self.ts.registerCallback(self.keyframe_callback)

        self.camera_info_sub = self.create_subscription(CameraInfo, '/camera/camera/infra2/camera_info', self.info_callback, 10)
        self.K = None
        self.baseline = None
        self.last_keyframe_image = None

        self.odom = {}
        self.pose_graph_used_pose = {}
        self.relative_pose_constraint = []
        self.last_keyframe_timestamp = None
        self.features = {}
        self.embeddings = {}

        self.loop_similarity_threshold = 0.90
        self.relocalization_threshold = 0.85
        self.loop_top_k = 1


        self.image_paths = {}
        self.depth_paths = {}
        self.mapping_mode = mapping_mode

        os.makedirs("tinynav_map/images", exist_ok=True)
        os.makedirs("tinynav_map/depths", exist_ok=True)

    def info_callback(self, msg:CameraInfo):
        if self.K is None:
            logging.info("Camera intrinsics received.")
            self.K = np.array(msg.k).reshape(3, 3)
            fx = self.K[0, 0]
            Tx = msg.p[3]
            self.baseline = -Tx / fx
            self.destroy_subscription(self.camera_info_sub)


    def keyframe_callback(self, keyframe_image_msg:Image, keyframe_odom_msg:Odometry, disp_msg:Image):
        if self.mapping_mode:
            self.keyframe_mapping(keyframe_image_msg, keyframe_odom_msg, disp_msg)
            keyframe_image_timestamp_ns = int(keyframe_image_msg.header.stamp.sec * 1e9) + int(keyframe_image_msg.header.stamp.nanosec)
            pose_graph_optimized_pose  = self.pose_graph_used_pose[keyframe_image_timestamp_ns]
            self.relocation_pub.publish(np2msg(pose_graph_optimized_pose, keyframe_image_msg.header.stamp, "world", "camera_frame"))
        else:
            image = self.bridge.imgmsg_to_cv2(keyframe_image_msg, desired_encoding="mono8")
            self.keyframe_relocalization(keyframe_image_msg.header.stamp, image)

    @Timer(name="Mapping Loop", text="\n\n[{name}] Elapsed time: {milliseconds:.0f} ms")
    def keyframe_mapping(self, keyframe_image_msg:Image, keyframe_odom_msg:Odometry, disp_msg:Image):
        if self.K is None:
            return
        keyframe_image_timestamp = int(keyframe_image_msg.header.stamp.sec * 1e9) + int(keyframe_image_msg.header.stamp.nanosec)
        keyframe_odom_timestamp = int(keyframe_odom_msg.header.stamp.sec * 1e9) + int(keyframe_odom_msg.header.stamp.nanosec)
        disp_timestamp = int(disp_msg.header.stamp.sec * 1e9) + int(disp_msg.header.stamp.nanosec)
        assert keyframe_image_timestamp == keyframe_odom_timestamp
        assert keyframe_image_timestamp == disp_timestamp
        disp = self.bridge.imgmsg_to_cv2(disp_msg, desired_encoding="32FC1")
        depth = self.compute_depth_from_disparity(disp, self.K, self.baseline)

        odom = msg2np(keyframe_odom_msg)

        image = self.bridge.imgmsg_to_cv2(keyframe_image_msg, desired_encoding="mono8")

        image_path = f"tinynav_map/images/{keyframe_image_timestamp}.npy"
        np.save(image_path, image)
        self.image_paths[keyframe_image_timestamp] = image_path

        depth_path = f"tinynav_map/depths/depth_{keyframe_image_timestamp}.npy"
        np.save(depth_path, depth)
        self.depth_paths[keyframe_image_timestamp] = depth_path

        self.embeddings[keyframe_image_timestamp] = self.get_embeddings(image)
        self.features[keyframe_image_timestamp]  = self.extract_super_point_features(image)

        if len(self.odom) == 0 and self.last_keyframe_timestamp is None:
            self.odom[keyframe_odom_timestamp] = odom
            self.pose_graph_used_pose[keyframe_odom_timestamp] = odom
            self.last_keyframe_timestamp = keyframe_odom_timestamp
        else:
            last_keyframe_odom_pose = self.odom[self.last_keyframe_timestamp]
            T_prev_curr = np.linalg.inv(last_keyframe_odom_pose) @ odom

            self.relative_pose_constraint.append((keyframe_image_timestamp, self.last_keyframe_timestamp, T_prev_curr, np.array([10.0, 10.0, 10.0]), np.array([10.0, 10.0, 10.0])))
            # debug used
            # prev_features = self.features[self.last_keyframe_timestamp]
            # curr_features = self.features[keyframe_image_timestamp]
            # prev_keypoints_origin = prev_features['kpts'].squeeze()
            # curr_keypoints_origin = curr_features['kpts'].squeeze()
            # prev_descriptors_origin = prev_features['descps'].squeeze()
            # curr_descriptors_origin = curr_features['descps'].squeeze()
            # prev_keypoints, curr_keypoints, matches = self.match_keypoints(prev_features, curr_features)
            # success, T_prev_curr_from_features, inliers_2d, inliers_3d, inliers = self.compute_relative_pose(depth, prev_keypoints, curr_keypoints, self.K)
            # matches_image = draw_image_match_origin(self.last_keyframe_image, image_rgb, prev_keypoints_origin, curr_keypoints_origin, matches)
            # self.publish_matches_image(keyframe_image_timestamp, matches_image)
            # if success:
                # projected_2d = project_3d_to_2d_into_image(cv2.cvtColor(self.last_keyframe_image, cv2.COLOR_GRAY2BGR), inliers_2d, inliers_3d, T_prev_curr_from_features, self.K)
                # self.publish_projected_3d_to_2d(keyframe_image_timestamp,projected_2d)
            self.pose_graph_used_pose[keyframe_image_timestamp] = odom
            self.odom[keyframe_image_timestamp] = odom
            with Timer(name = "pose graph", text="[{name}] Elapsed time: {milliseconds:.0f} ms"):
                self.pose_graph_solve()
            with Timer(name = "loop", text="[{name}] Elapsed time: {milliseconds:.0f} ms"):
                self.find_loop(keyframe_image_timestamp, depth)
            self.pose_graph_trajectory_publish(keyframe_image_timestamp)
            self.last_keyframe_timestamp = keyframe_odom_timestamp
        self.last_keyframe_image = image


    def pose_graph_solve(self):
        """
        Solve the bundle adjustment problem.
        """
        min_timestamp = min(self.pose_graph_used_pose.keys())
        constant_pose_index_dict = { min_timestamp : True }
        optimized_camera_poses = pose_graph_solve(self.pose_graph_used_pose, self.relative_pose_constraint, constant_pose_index_dict)
        for timestamp, pose in optimized_camera_poses.items():
            self.pose_graph_used_pose[timestamp] = pose
        return optimized_camera_poses


    def compute_relative_pose(self, depth_curr:np.ndarray, kps_prev:np.ndarray, kps_curr:np.ndarray, K: np.ndarray) -> tuple[bool, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        points_3d, points_2d = [], []
        for kp_prev, kp_curr in zip(kps_prev, kps_curr):
            u, v = int(kp_curr[0]), int(kp_curr[1])
            if depth_curr[v, u] > 0 and depth_curr[v, u] < 50:
                Z = depth_curr[v, u]
                X = (kp_curr[0] - K[0, 2]) * Z / K[0, 0]
                Y = (kp_curr[1] - K[1, 2]) * Z / K[1, 1]
                points_3d.append(np.array([X, Y, Z]))
                points_2d.append(np.array([kp_prev[0], kp_prev[1]]))
        if len(points_3d) < 20:
            # log the warning info
            logging.warning("Not enough points to compute relative pose")
            return False, np.eye(4), None, None, None
        points_3d = np.array(points_3d)
        points_2d = np.array(points_2d)
        # === Solve PnP
        success, rvec, tvec, inliers = cv2.solvePnPRansac(points_3d, points_2d, K, None)
        if not success:
            # log the warning info
            logging.warning("Warning: Failed to solve PnP")
            return False, np.eye(4), None, None, None
        if len(inliers) < 20:
            # log the warning info
            logging.warning(f"Not enough inliers[{len(inliers)} < 20] to compute relative pose")
            return False, np.eye(4), None, None, None
        R, _ = cv2.Rodrigues(rvec)
        T_prev_curr = np.eye(4)
        T_prev_curr[:3, :3] = R
        T_prev_curr[:3, 3] = tvec.ravel()
        inliers = inliers.flatten()
        inliers_2d = points_2d[inliers]
        inliers_3d = points_3d[inliers]
        return True, T_prev_curr, inliers_2d, inliers_3d, inliers


    def compute_depth_from_disparity(self, disparity:np.ndarray, K:np.ndarray, baseline:float) -> np.ndarray:
        depth = np.zeros_like(disparity)
        depth[disparity > 0] = K[0, 0] * baseline / (disparity[disparity > 0])
        return depth

    def get_embeddings(self, image: np.ndarray) -> np.ndarray:
        processed_image = self.dinov2_model.preprocess_image(image)
        # shape: (1, 768)
        return self.dinov2_model.infer(processed_image)["last_hidden_state"][:, 0, :].squeeze(0)

    def extract_super_point_features(self, image: np.ndarray) -> dict:
        return self.super_point_extractor.infer(image)

    def match_keypoints(self, feats0:dict, feats1:dict, image_shape = np.array([640, 360], dtype = np.int64)) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        match_result = self.light_glue_matcher.infer(feats0["kpts"], feats1["kpts"], feats0['descps'], feats1['descps'], image_shape, image_shape)
        match_indices = match_result["match_indices"][0]
        valid_mask = match_indices != -1
        keypoints0 = feats0["kpts"][0][valid_mask]
        keypoints1 = feats1["kpts"][0][match_indices[valid_mask]]
        matches = []
        for i, index in enumerate(match_indices):
            if index != -1:
                matches.append([i, index])
        return keypoints0, keypoints1, np.array(matches, dtype=np.int64)

    def pose_graph_trajectory_publish(self, timestamp):
        path_msg = Path()
        path_msg.header.stamp.sec = int(timestamp / 1e9)
        path_msg.header.stamp.nanosec = int(timestamp % 1e9)
        path_msg.header.frame_id = "world"
        for t, pose_in_world in self.pose_graph_used_pose.items():
            pose = PoseStamped()
            pose.header = path_msg.header
            t = pose_in_world[:3, 3]
            quat = matrix_to_quat(pose_in_world[:3, :3])
            pose.pose.position.x = t[0]
            pose.pose.position.y = t[1]
            pose.pose.position.z = t[2]
            pose.pose.orientation.x = quat[0]
            pose.pose.orientation.y = quat[1]
            pose.pose.orientation.z = quat[2]
            pose.pose.orientation.w = quat[3]
            path_msg.poses.append(pose)
        self.pose_graph_trajectory_pub.publish(path_msg)

    def publish_projected_3d_to_2d(self, timestamp:int, projected_2d:np.ndarray):
        msg = self.bridge.cv2_to_imgmsg(projected_2d, encoding="bgr8")
        msg.header.stamp.sec = int(timestamp / 1e9)
        msg.header.stamp.nanosec = int(timestamp % 1e9)
        msg.header.frame_id = "camera_frame"
        self.project_3d_to_2d_pub.publish(msg)

    def publish_matches_image(self, timestamp: int, matches_image:np.ndarray):
        msg = self.bridge.cv2_to_imgmsg(matches_image, encoding="bgr8")
        msg.header.stamp.sec = int(timestamp / 1e9)
        msg.header.stamp.nanosec = int(timestamp % 1e9)
        msg.header.frame_id = "camera_frame"
        self.matches_image_pub.publish(msg)

    def publish_loop_matches_image(self, timestamp: int, matches_image:np.ndarray):
        msg = self.bridge.cv2_to_imgmsg(matches_image, encoding="bgr8")
        msg.header.stamp.sec = int(timestamp / 1e9)
        msg.header.stamp.nanosec = int(timestamp % 1e9)
        msg.header.frame_id = "camera_frame"
        self.loop_matches_image_pub.publish(msg)

    def find_loop(self, timestamp: int, depth_image:np.ndarray):
        all_timestamps_ = list(self.pose_graph_used_pose.keys())
        all_timestamps_.sort()
        valid_timestamps = set([t for t in all_timestamps_ if t + 10 * 1e9 < timestamp])
        query_embedding = self.embeddings[timestamp]
        query_embedding_normed = query_embedding / np.linalg.norm(query_embedding)
        max_similarity = 0
        similarity_array = []
        for prev_timestamp in valid_timestamps:
            prev_embedding = self.embeddings[prev_timestamp]
            prev_embedding_normed = prev_embedding / np.linalg.norm(prev_embedding)
            similarity = np.dot(query_embedding_normed, prev_embedding_normed.T).item()
            if similarity > max_similarity:
                max_similarity = similarity
            if similarity > self.loop_similarity_threshold:
                similarity_array.append((prev_timestamp, similarity))
        # sort similarity_array by similarity, larger first
        similarity_array.sort(key=lambda x: x[1], reverse=True)
        if len(similarity_array) > 0:
                query_features = self.features[timestamp]
                query_keypoints_origin = query_features['kpts'].squeeze()
                _ = query_features['descps'].squeeze()
                for prev_timestamp, similarity in similarity_array[:self.loop_top_k]:
                    reference_features = self.features[prev_timestamp]
                    reference_keypoints_origin = reference_features['kpts'].squeeze()
                    _ = reference_features['descps'].squeeze()
                    reference_matched_keypoints, query_matched_keypoints, matches = self.match_keypoints(reference_features, query_features)

                    query_image = np.load(self.image_paths[timestamp])
                    reference_image = np.load(self.image_paths[prev_timestamp])
                    query_image_rgb = cv2.cvtColor(query_image, cv2.COLOR_GRAY2RGB)
                    reference_image_rgb = cv2.cvtColor(reference_image, cv2.COLOR_GRAY2RGB)
                    image_match_image = draw_image_match_origin(reference_image_rgb, query_image_rgb, reference_keypoints_origin, query_keypoints_origin, matches)
                    cv2.putText(image_match_image, f"similarity: {similarity:.2f}, query_keypoints: {len(query_keypoints_origin)}, reference_keypoints: {len(reference_keypoints_origin)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.imwrite(f"map/loop_matches_{timestamp}_{prev_timestamp}.png", image_match_image)
                    self.publish_loop_matches_image(timestamp, image_match_image)
                    success, T_prev_curr_from_features, inliers_2d, inliers_3d, inliers = self.compute_relative_pose(depth_image, reference_matched_keypoints, query_matched_keypoints, self.K)
                    if success:
                        if len(inliers) >= 100:
                            self.relative_pose_constraint.append((timestamp, prev_timestamp, T_prev_curr_from_features, np.array([10.0, 10.0, 10.0]), np.array([10.0, 10.0, 10.0])))
                            logging.info(f"loop detected with similarity {similarity:.2f}, inliers: {len(inliers)}")
                        else:
                            logging.info(f"similarity {similarity:.2f} not enough inliers to relocalize, {len(inliers)} / {len(matches)}")
                    else:
                        logging.info(f"similarity {similarity:.2f} failed to compute relative pose")
        else:
            logging.info(f"not enough similar embeddings to find loop, {len(similarity_array)}, max_similarity : {max_similarity}")


    def relocalize_with_depth(self, keyframe: np.ndarray, keyframe_features: dict, K: np.ndarray) -> tuple[bool, np.ndarray]:
        query_embedding = self.get_embeddings(keyframe)
        query_embedding_normed = query_embedding / np.linalg.norm(query_embedding)

        total_timestamps = list(self.embeddings.keys())
        total_timestamps.sort()

        similarity_array = []
        max_similarity = 0
        for prev_timestamp in total_timestamps:
            prev_embedding = self.embeddings[prev_timestamp]
            prev_embedding_normed = prev_embedding / np.linalg.norm(prev_embedding)
            similarity = np.dot(query_embedding_normed, prev_embedding_normed.T).item()
            if similarity > max_similarity:
                max_similarity = similarity
            if similarity > self.relocalization_threshold:
                similarity_array.append((prev_timestamp, similarity))
        # sort similarity_array by similarity, larger first
        similarity_array.sort(key=lambda x: x[1], reverse=True)

        if len(similarity_array) > 0:
            point_3d_in_world_list = []
            point_2d_in_keyframe_list = []

            for prev_timestamp, similarity in similarity_array[:self.loop_top_k]:
                reference_keyframe_pose = self.pose_graph_used_pose[prev_timestamp]
                reference_features = self.features[prev_timestamp]
                reference_depth = np.load(self.depth_paths[prev_timestamp])

                reference_matched_keypoints, keyframe_matched_keypoints, matches = self.match_keypoints(reference_features, keyframe_features)
                if len(matches) > 20:
                    point_3d_in_world, inliers = self.keypoint_with_depth_to_3d(reference_matched_keypoints, reference_depth, reference_keyframe_pose, self.K)
                    point_3d_in_world_list = point_3d_in_world[inliers]
                    point_2d_in_keyframe_list = keyframe_matched_keypoints[inliers]
                else:
                    print(f"not enough matched features to relocalize, {len(point_3d_in_world_list)}")
                    return False, np.eye(4)

            if len(point_3d_in_world_list) > 20:
                point_3d_in_world_list = np.array(point_3d_in_world_list)
                point_2d_in_keyframe_list = np.array(point_2d_in_keyframe_list)

                success, rvec, tvec, inliers = cv2.solvePnPRansac(point_3d_in_world_list, point_2d_in_keyframe_list, K, None)
                if success and len(inliers) >= 20:
                    R, _ = cv2.Rodrigues(rvec)
                    T = np.eye(4)
                    T[:3, :3] = R
                    T[:3, 3] = tvec.reshape(3)
                    print(f"relocalization pose : {T}")
                    return True, T
            else:
                print(f"not enough landmarks to relocalize, {len(point_3d_in_world_list)}")
                return False, np.eye(4)
        else:
            print(f"not enough similar embeddings to relocalize, {len(similarity_array)}, max_similarity : {max_similarity}")
        return False, np.eye(4)

    def keypoint_with_depth_to_3d(self, keypoints:np.ndarray, depth:np.ndarray, pose_from_camera_to_world:np.ndarray, K:np.ndarray):
        point_in_camera = []
        inliers = []
        fx = K[0, 0]
        fy = K[1, 1]
        cx = K[0, 2]
        cy = K[1, 2]
        for kp in keypoints:
            u = int(kp[0])
            v = int(kp[1])
            Z = depth[v, u]
            if Z > 0 and Z < 50:
                X = (u - cx) * Z / fx
                Y = (v - cy) * Z / fy
                inliers.append(True)
            else:
                X = 0
                Y = 0
                inliers.append(False)
            point_in_camera.append(np.array([X, Y, Z]))
        # shape: (N, 3)
        point_in_camera = np.array(point_in_camera)
        inliers = np.array(inliers)
        rotation = pose_from_camera_to_world[:3, :3]
        translation = pose_from_camera_to_world[:3,3]

        point_in_world = (rotation @ point_in_camera.T).T + translation
        return point_in_world, inliers

    @Timer(name="Relocalization loop", text="\n\n[{name}] Elapsed time: {milliseconds:.0f} ms")
    def keyframe_relocalization(self, timestamp, image:np.ndarray) -> tuple[bool, np.ndarray]:
        features = self.extract_super_point_features(image)
        res, pose_in_camera = self.relocalize_with_depth(image, features, self.K)
        if res:
            pose_in_world = np.linalg.inv(pose_in_camera)
            self.relocation_pub.publish(np2msg(pose_in_world, timestamp, "world", "camera"))

    def load_mapping(self):
        self.embeddings = np.load("tinynav_map/embedding.npy", allow_pickle=True).item()
        self.features = np.load("tinynav_map/features.npy", allow_pickle=True).item()
        self.depth_paths = np.load("tinynav_map/depth_paths.npy", allow_pickle=True).item()
        self.pose_graph_used_pose = np.load("tinynav_map/poses.npy", allow_pickle=True).item()
        self.K = np.load("tinynav_map/intrinsics.npy", allow_pickle=True)
        logging.info("load map from mapping directory")

        # genereate point cloud from the depths
        self.point_cloud = self.genereate_point_cloud_from_depths()

    def save_mapping(self):
        if not os.path.exists("tinynav_map"):
            os.mkdir("tinynav_map")
        # self.landmark_tracker.save_to_dir("mapping")
        np.save("tinynav_map/embedding.npy", self.embeddings)
        np.save("tinynav_map/features.npy", self.features)
        np.save("tinynav_map/depth_paths.npy", self.depth_paths, allow_pickle=True)
        np.save("tinynav_map/poses.npy", self.pose_graph_used_pose)
        np.save("tinynav_map/intrinsics.npy", self.K)
        logging.info("save map into mapping directory")
        self.generate_occupancy_map()

    def on_shutdown(self):
        if self.mapping_mode:
            self.save_mapping()

    def genereate_point_cloud_from_depths(self):
        point_clouds = []
        for timestamp, depth_path in self.depth_paths.items():
            depth = np.load(depth_path)
            point_cloud_in_camera = depth_to_cloud(depth, self.K)
            # downsample the point cloud to reduce the number of points
            point_cloud_in_camera = point_cloud_in_camera[::24]
            # filter point_cloud distance larger then 3m
            dist = np.linalg.norm(point_cloud_in_camera, axis=1)
            valid_indices = dist < 3.0
            point_cloud_in_camera_filtered = point_cloud_in_camera[valid_indices]
            pose = self.pose_graph_used_pose[timestamp]
            # transform point cloud to world frame
            point_cloud_in_world = transform_point_cloud(point_cloud_in_camera_filtered, pose)
            point_clouds.append(point_cloud_in_world)

        if len(point_clouds):
            return np.vstack(point_clouds)
        else:
            print("No point clouds found in depth paths")
            return None

    def global_map_callback(self):
        if not self.mapping_mode and self.point_cloud is not None:
            self.publish_global_map(self.point_cloud)
        else:
            point_clouds = self.genereate_point_cloud_from_depths()
            if point_clouds is not None:
                self.publish_global_map(point_clouds)

    def publish_global_map(self, point_cloud:np.ndarray):
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = 'world'
        cloud = pc2.create_cloud_xyz32(header, point_cloud.tolist())
        self.global_map_pub.publish(cloud)
        print(f"Published global map with {len(point_cloud)} points")

    def generate_occupancy_map(self):
        """
            Genereate a occupancy grid map from the depth images.
            The occupancy grid map is a 3D grid with the following values:
             0 : Unknown
             1 : Free
             2 : Occupied
        """
        raycast_shape = (100, 100, 20)
        resolution = 0.05
        step = 10
        fx, fy, cx, cy = self.K[0, 0], self.K[1, 1], self.K[0, 2], self.K[1, 2]
        global_grid = None
        global_origin = None
        for timestamp, odom_pose in self.pose_graph_used_pose.items():
            depth = np.load(self.depth_paths[timestamp])
            odom_translation = odom_pose[:3, 3]
            local_origin = odom_translation - 0.5 * np.array(raycast_shape) * resolution
            local_grid = run_raycasting_loopy(depth, odom_pose, raycast_shape, fx, fy, cx, cy, local_origin, step, resolution, True)
            if global_grid is None:
                global_grid = local_grid
                global_origin = local_origin
            else:
                global_grid, global_origin = merge_grids(global_grid, global_origin, local_grid, local_origin, resolution)
        grid_type = np.zeros_like(global_grid, dtype=np.uint8)
        grid_type[global_grid > 0] = 2  # Occupied
        grid_type[global_grid < 0] = 1  # Free
        x_y_plane = np.max(grid_type, axis=2)
        print(f"x_y_plane type counts - Occupied: {np.sum(x_y_plane == 2)}, Free: {np.sum(x_y_plane == 1)}, Unknown: {np.sum(x_y_plane == 0)}")
        # remapping type to 0 - 255 for display
        x_y_plane_image = np.zeros_like(x_y_plane, dtype=np.float32)
        x_y_plane_image[x_y_plane == 2] = 1.0
        x_y_plane_image[x_y_plane == 1] = 0.5
        x_y_plane_image = (x_y_plane_image * 255).astype(np.uint8)
        cv2.imwrite("tinynav_map/occupancy_map.png", x_y_plane_image)
        meta_info = np.array([global_origin[0], global_origin[1], resolution], dtype=np.float32)
        np.save("tinynav_map/occupancy_map_meta.npy", meta_info)
        np.save("tinynav_map/occupancy_map.npy", grid_type)
        print("occupancy map generated")



def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")

def main(args=None):
    rclpy.init(args=args)
    parser = argparse.ArgumentParser()
    parser.add_argument("--mapping", default=False, type=str2bool)
    parsed_args, unknown_args = parser.parse_known_args(sys.argv[1:])
    print(f"Map Node Mode : {parsed_args.mapping}")
    node = MapNode(mapping_mode=parsed_args.mapping)
    if not parsed_args.mapping:
        node.load_mapping()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if parsed_args.mapping:
            node.on_shutdown()

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
