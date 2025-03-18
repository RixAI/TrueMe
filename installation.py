
import os
import sys

# Check if running in Colab
IN_COLAB = 'google.colab' in sys.modules

def run_command(cmd):
    if IN_COLAB:
        os.system(f"source /content/venv/bin/activate && {cmd}")
    else:
        os.system(cmd)

# Update pip
run_command("pip install --upgrade pip")

# Install dependencies
with open('requirements.txt', 'w') as req:
    req.write("""
numpy==1.24.3
opencv-python==4.8.0.74
onnx==1.14.0
insightface==0.7.3
onnxruntime-gpu==1.21.0
tqdm==4.65.0
moviepy==1.0.3
torch==2.0.1+cu118
""")
run_command("pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu118 --no-cache-dir")

# Download models
MODEL_DIR = "./models"
os.makedirs(MODEL_DIR, exist_ok=True)
if not os.path.exists(f"{MODEL_DIR}/inswapper_128.onnx"):
    os.system(f"gdown --id 1vd7QgByf5QXFiqYrEmuHhHc_4QfL90SD -O {MODEL_DIR}/inswapper_128.onnx")
if not os.path.exists(f"{MODEL_DIR}/buffalo_l"):
    os.system(f"gdown --id 1Vym6zN9tH5oZ0rjWZ3oXHMR4ex9kIQxl -O {MODEL_DIR}/buffalo_l.zip && unzip -q {MODEL_DIR}/buffalo_l.zip -d {MODEL_DIR}")

print("Installation complete! Models downloaded to './models/'")
