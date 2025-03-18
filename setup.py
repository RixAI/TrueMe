
import os
import torch
import onnxruntime as ort
from insightface.app import FaceAnalysis

# Paths
MODEL_DIR = "./models"

# Check CUDA
use_cuda = torch.cuda.is_available()
print(f"CUDA Available: {use_cuda}")

# Load models
providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if use_cuda else ['CPUExecutionProvider']
try:
    face_analyzer = FaceAnalysis(name="buffalo_l", root=MODEL_DIR, providers=providers)
    face_analyzer.prepare(ctx_id=0 if use_cuda else -1, det_size=(640, 640))
    swapper = ort.InferenceSession(f"{MODEL_DIR}/inswapper_128.onnx", providers=providers)
    print("Models loaded successfully! Everything is connected.")
except Exception as e:
    print(f"Error loading models: {e}")
