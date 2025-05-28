import numpy as np
from codetiming import Timer
from tinynav.core.models_trt import SuperPointTRT

def test_superpoint_trt_with_cache():
    superpoint = SuperPointTRT("tinynav/models/superpoint_360x640_fp16.plan")
    # Create dummy zero inputs
    image_shape = np.array([640, 360], dtype=np.int64)
    width, height = image_shape

    dummy_image = np.random.randint(0, 256, (height, width, 1), dtype=np.uint8)
    dummy_image2 = np.random.randint(0, 256, (height, width, 1), dtype=np.uint8)

    with Timer(text="[superpoint] Elapsed time: {milliseconds:.0f} ms"):
        extract_result = superpoint.infer(dummy_image)

    with Timer(text="[superpoint] Elapsed time: {milliseconds:.0f} ms"):
        extract_result_origin = superpoint.infer(dummy_image2)

    with Timer(text="[superpoint] Elapsed time: {milliseconds:.0f} ms"):
        extract_result_first = superpoint.memorized_infer(dummy_image2)

    with Timer(text="[superpoint] Elapsed time: {milliseconds:.0f} ms"):
        extract_result_second = superpoint.memorized_infer(dummy_image2)
    assert np.array_equal(extract_result_origin['kpts'], extract_result_first['kpts']), "Cached first kpts result does not match original result."
    assert np.array_equal(extract_result_origin['descps'], extract_result_first['descps']), "Cached first descps result does not match original result."
    assert np.array_equal(extract_result_origin['kpts'], extract_result_second['kpts']), "Cached second kpts result does not match original result."
    assert np.array_equal(extract_result_origin['descps'], extract_result_second['descps']), "Cached second descps result does not match original result."

if __name__ == "__main__":
    test_superpoint_trt_with_cache()
    print("SuperPoint TRT with cache test passed.")

