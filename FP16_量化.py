import onnx
from onnxconverter_common import float16
 
model = onnx.load("runs\\train\\exp2\\weights\\best.onnx")
model_fp16 = float16.convert_float_to_float16(model)
onnx.save(model_fp16,"runs\\train\\exp2\\weights\\best_fp16.onnx")