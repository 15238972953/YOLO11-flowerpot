import onnx
# from onnxruntime.quantization import quantize_qat, QuantType
from onnxruntime.quantization import quantize_dynamic, QuantType,quantize_static

model_fp32 = "runs\\train\\exp2\\weights\\best.onnx"
model_quant = "runs\\train\\exp2\\weights\\best_int8.onnx"
 
# 加载FP32模型
onnx_model = onnx.load(model_fp32)
 
# 进行量化
quantized_model = quantize_dynamic(model_fp32, model_quant)
# quantized_model = quantize_qat(
#     model=onnx_model,
#     quantization_type=QuantType.QInt8,
#     force_fusions=True
# )
 
# 保存量化模型
onnx.save_model(quantized_model, model_quant)