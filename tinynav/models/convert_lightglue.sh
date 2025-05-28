trtexec --onnx=./lightglue_v1.onnx \
        --inputIOFormats="fp32:chw","fp32:chw","uint8:chw","uint8:chw","int32:chw","int32:chw","fp32:chw" \
                --minShapes='kpts0':1x1x2,'kpts1':1x1x2,'desc0':1x1x256,'desc1':1x1x256,'img_shape0':1x2,'img_shape1':1x2 \
                --optShapes='kpts0':1x1535x2,'kpts1':1x1546x2,'desc0':1x1535x256,'desc1':1x1546x256,'img_shape0':1x2,'img_shape1':1x2 \
                --maxShapes='kpts0':1x2048x2,'kpts1':1x2048x2,'desc0':1x2048x256,'desc1':1x2048x256,'img_shape0':1x2,'img_shape1':1x2 \
        --fp16 \
        --precisionConstraints=obey \
        --layerPrecisions="/backbone/self_attn.0/inner_attn/Einsum:fp16","/backbone/self_attn.0/inner_attn_1/Einsum:fp16" \
        --saveEngine=./lightglue_fp16_jetson.plan
