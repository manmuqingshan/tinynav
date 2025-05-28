#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <Eigen/Dense>
#include <vector>
#include <algorithm>
#include <cmath>

namespace py = pybind11;

Eigen::MatrixXd run_raycasting_cpp(
    py::array_t<float, py::array::c_style | py::array::forcecast> depth_image,
    Eigen::Matrix<double, 4, 4, Eigen::RowMajor> T_cam_to_world,
    std::vector<int> grid_shape,
    double fx, double fy, double cx, double cy,
    Eigen::Vector3d origin,
    int step,
    double resolution) {

    py::buffer_info buf = depth_image.request();
    auto* ptr = static_cast<float*>(buf.ptr);
    int depth_height = buf.shape[0];
    int depth_width = buf.shape[1];

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> occupancy_grid = Eigen::MatrixXd::Zero(grid_shape[0], grid_shape[1] * grid_shape[2]);
    occupancy_grid.resize(grid_shape[0], grid_shape[1] * grid_shape[2]);
    Eigen::Map<Eigen::VectorXd> occupancy_grid_flat(occupancy_grid.data(), occupancy_grid.size());


    Eigen::Vector3d camera_origin = T_cam_to_world.topRightCorner(3, 1);
    Eigen::Vector3i start_voxel_base = ((camera_origin - origin) / resolution).array().floor().cast<int>();

    for (int v = 0; v < depth_height; v += step) {
        for (int u = 0; u < depth_width; u += step) {
            float d = ptr[v * depth_width + u];
            if (d <= 0) {
                continue;
            }

            double x = (u - cx) * d / fx;
            double y = (v - cy) * d / fy;
            double z = d;

            Eigen::Vector4d point_cam(x, y, z, 1.0);
            Eigen::Vector4d point_world_h = T_cam_to_world * point_cam;
            Eigen::Vector3d point_world = point_world_h.head<3>();

            Eigen::Vector3i start_voxel = start_voxel_base;
            Eigen::Vector3i end_voxel = ((point_world - origin) / resolution).array().floor().cast<int>();
            Eigen::Vector3i diff = end_voxel - start_voxel;
            int steps = diff.cwiseAbs().maxCoeff();

            if (steps == 0) {
                continue;
            }

            for (int i = 0; i <= steps; ++i) {
                double t = static_cast<double>(i) / steps;
                Eigen::Vector3i interp = (start_voxel.cast<double>() + t * diff.cast<double>()).unaryExpr([](double v) { return std::nearbyint(v); }).cast<int>();

                if ((interp.array() < 0).any() || (interp.array() >= Eigen::Map<Eigen::VectorXi>(grid_shape.data(), 3).array()).any()) {
                    continue;
                }
                int flat_idx = interp.x() * (grid_shape[1] * grid_shape[2]) + interp.y() * grid_shape[2] + interp.z();
                occupancy_grid_flat(flat_idx) -= 0.05;
            }

            if ((end_voxel.array() >= 0).all() && (end_voxel.array() < Eigen::Map<Eigen::VectorXi>(grid_shape.data(), 3).array()).all()) {
                 int flat_idx = end_voxel.x() * (grid_shape[1] * grid_shape[2]) + end_voxel.y() * grid_shape[2] + end_voxel.z();
                occupancy_grid_flat(flat_idx) += 0.2;
            }
        }
    }
    
    occupancy_grid_flat = occupancy_grid_flat.cwiseMax(-0.1).cwiseMin(0.1);
    
    // The map should have modified the underlying data of occupancy_grid.
    // The reshape will be done in python
    return occupancy_grid;
}


PYBIND11_MODULE(raycast_cpp, m) {
    m.doc() = "C++ implementation of run_raycasting";
    m.def("run_raycasting_cpp", &run_raycasting_cpp, "A function that performs raycasting on a depth image",
          py::arg("depth_image"),
          py::arg("T_cam_to_world"),
          py::arg("grid_shape"),
          py::arg("fx"), py::arg("fy"), py::arg("cx"), py::arg("cy"),
          py::arg("origin"),
          py::arg("step"),
          py::arg("resolution"));
}
