"""
stereo_engine.py
A fast stereo inference engine using TensorRT and PyCUDA, as prototyped in foundation_stereo.ipynb.
"""
import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # noqa: F401

# D535
# input (480, 848) -> (181, 320)
# model input (288, 320)

# D455
# input (360, 640) -> (180, 320)
# model input (288, 320)

class StereoEngine:
    def __init__(self, engine_file_path='/tinynav/tinynav/models/foundation_stereo_11-33-40_288x320.plan'):
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()
        self.inputs = []
        self.outputs = []
        for idx in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(idx)
            mode = self.engine.get_tensor_mode(name)
            shape = self.engine.get_tensor_shape(name)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            if -1 in shape:
                pass
            vol = int(np.prod(shape).item())
            host_mem = cuda.pagelocked_empty(vol, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            tensor = {
                'name': name,
                'mode': mode,
                'shape': shape,
                'dtype': dtype,
                'host': host_mem,
                'device': device_mem
            }
            if mode == trt.TensorIOMode.INPUT:
                self.inputs.append(tensor)
            else:
                self.outputs.append(tensor)

    def infer(self, left_img, right_img):
        # Convert mono8 to RGB if needed
        if len(left_img.shape) == 2:
            left_img = cv2.cvtColor(left_img, cv2.COLOR_GRAY2RGB)
        if len(right_img.shape) == 2:
            right_img = cv2.cvtColor(right_img, cv2.COLOR_GRAY2RGB)

        # Check input shapes
        if left_img.shape == (480, 848, 3):
            self.im_shape = (480, 848)  # Original image shape (height, width)
            self.mid_shape = (181, 320)
            self.input_shape = (288, 320)
            self.v_shift = int((self.input_shape[0] - self.mid_shape[0]) / 2)  # Vertical shift to center the resized image
        elif left_img.shape == (360, 640, 3):
            self.im_shape = (360, 640)  # Original image shape (height, width)
            self.mid_shape = (180, 320)
            self.input_shape = (288, 320)
            self.v_shift = int((self.input_shape[0] - self.mid_shape[0]) / 2)  # Vertical shift to center the resized image
        else:
            raise ValueError(f"Unsupported input shape: {left_img.shape}. Expected (480, 848) or (360, 640).")

        # Resize to engine input size
        def resize_and_fill(img):
            resized = cv2.resize(img, (self.mid_shape[1], self.mid_shape[0]), interpolation=cv2.INTER_LINEAR)
            filled = np.zeros((self.input_shape[0], self.input_shape[1], 3), dtype=np.uint8)
            filled[self.v_shift:self.v_shift + resized.shape[0], :resized.shape[1]] = resized
            return filled

        left_resized = resize_and_fill(left_img)
        right_resized = resize_and_fill(right_img)
        # Convert to CHW float32
        left_tensor = np.transpose(left_resized, (2, 0, 1)).astype(np.float32)
        right_tensor = np.transpose(right_resized, (2, 0, 1)).astype(np.float32)
        # If dynamic shape: set shapes
        for tensor in self.inputs:
            if -1 in tensor['shape']:
                shape = (1,) + left_tensor.shape
                self.context.set_input_shape(tensor['name'], shape)
                tensor['shape'] = shape
                tensor['vol'] = int(np.prod(shape))
        # Fill input data
        for inp in self.inputs:
            if 'left' in inp['name'].lower():
                inp['host'][:] = left_tensor.flatten()
            elif 'right' in inp['name'].lower():
                inp['host'][:] = right_tensor.flatten()
        # Upload inputs
        for inp in self.inputs:
            cuda.memcpy_htod_async(inp['device'], inp['host'], self.stream)
        # Bind tensors
        for inp in self.inputs:
            self.context.set_tensor_address(inp['name'], int(inp['device']))
        for out in self.outputs:
            self.context.set_tensor_address(out['name'], int(out['device']))
        start = cuda.Event()
        end = cuda.Event()
        start.record(stream=self.stream)
        self.context.execute_async_v3(stream_handle=self.stream.handle)
        end.record(stream=self.stream)
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out['host'], out['device'], self.stream)
        self.stream.synchronize()
        gpu_time = start.time_till(end)
        results = []
        for out in self.outputs:
            # Output shape: (1, 1, H, W)
            out_map = out['host'].reshape(out['shape'])[0,0,:,:]

            # remove_invisible
            yy,xx = np.meshgrid(np.arange(out_map.shape[0]), np.arange(out_map.shape[1]), indexing='ij')
            us_right = xx - out_map
            invalid = us_right < 0
            out_map[invalid] = np.inf

            # Resize to original size and scale if needed
            #out_map_resized = cv2.resize(out_map, (original_size[1], original_size[0]), interpolation=cv2.INTER_LINEAR)
            # Optionally scale disparity if model output is normalized
            out_map_cropped = out_map[self.v_shift:self.v_shift + self.mid_shape[0], :]
            out_map_resized = cv2.resize(out_map_cropped, (self.im_shape[1], self.im_shape[0])) * (self.im_shape[1] / self.mid_shape[1])
            results.append(out_map_resized.astype(np.float32))
        return results, gpu_time

if __name__ == "__main__":
    engine = StereoEngine()

    # Create dummy zero inputs
    height, width = (480, 848)
    dummy_left = np.zeros((height, width, 3), dtype=np.uint8)
    dummy_right = np.zeros((height, width, 3), dtype=np.uint8)

    # Run inference
    results, gpu_time_ms = engine.infer(dummy_left, dummy_right)

    print(f"GPU inference time: {gpu_time_ms:.2f} ms")

