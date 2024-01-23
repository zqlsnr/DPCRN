# DPCRN
real-time speech enhance denoise C++ python tflite onnx

# Requirements
tensorflow==2.3.0
numpy==1.18.5
librosa==0.7.0
sondfile==0.10.3.post1
tflite_runtime==2.5.0
onnxruntime==1.6.0
onnxmltools==1.11.0

# Result

test dataset: 2020 no_reverb_16000
dpcrn_quant.tflite
    pesq: 3.41
    stoi: 96.26
dpcrn_quant_20epochs.tflite
    pesq: 3.485
    stoi: 96.81


