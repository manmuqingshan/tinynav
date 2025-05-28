trtexec --onnx=./superpoint.onnx \
                --minShapes='image':1x360x640x1,'keypoint_threshold':1x1 \
                --optShapes='image':1x360x640x1,'keypoint_threshold':1x1 \
                --maxShapes='image':1x360x640x1,'keypoint_threshold':1x1 \
                --device=0 \
        --fp16 \
        --saveEngine=./superpoint_fp16_jetson.plan
