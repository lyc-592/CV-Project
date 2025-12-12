import torch
import torch.nn as nn
from model import C_DCE_Net
import os
import subprocess


class DeployModel(nn.Module):
    def __init__(self, original_model):
        super(DeployModel, self).__init__()
        self.model = original_model

    def forward(self, x):
        enhanced, _ = self.model(x)
        return enhanced


def export():
    weights_path = "weights/ZeroDCE_final.pth"
    onnx_path = "weights/ZeroDCE.onnx"
    output_folder = "weights/tflite_output"

    if not os.path.exists(weights_path):
        print(f"Error: {weights_path} not found. Run train.py first!")
        return

    print(">>> Loading PyTorch model...")
    model = C_DCE_Net()
    # 这一步是为了防止版本差异导致的权重键名不匹配，加了 strict=False
    try:
        model.load_state_dict(torch.load(weights_path, map_location='cpu'))
    except:
        print("Warning: Loading weights strictly failed, trying strict=False...")
        model.load_state_dict(torch.load(weights_path, map_location='cpu'), strict=False)

    model.eval()
    deploy_model = DeployModel(model)

    print(">>> Exporting to ONNX...")
    dummy_input = torch.randn(1, 3, 256, 256)

    # 【核心】PyTorch 2.2.0 会完美支持 opset_version=11
    torch.onnx.export(
        deploy_model,
        dummy_input,
        onnx_path,
        verbose=False,
        input_names=['input'],
        output_names=['output'],
        opset_version=11
    )
    print(f"ONNX exported to {onnx_path}")

    # 恢复 onnxsim，PyTorch 2.2.0 导出的模型可以被完美简化
    print(">>> Simplifying ONNX model...")
    try:
        import onnxsim
        import onnx
        onnx_model = onnx.load(onnx_path)
        model_simp, check = onnxsim.simplify(onnx_model)
        onnx.save(model_simp, onnx_path)
        print("ONNX simplified successfully.")
    except Exception as e:
        print(f"Warning: onnxsim failed ({e}), using original ONNX...")

    print(">>> Converting to TFLite using onnx2tf...")
    # 此时 onnx2tf 不管是 1.20 还是 1.25 都能完美处理 Opset 11
    cmd = f"onnx2tf -i {onnx_path} -o {output_folder}"

    process = subprocess.Popen(cmd, shell=True)
    process.wait()

    if process.returncode == 0:
        print("\n" + "=" * 40)
        print("🎉 SUCCESS! Conversion finished.")
        if os.path.exists(output_folder):
            for f in os.listdir(output_folder):
                if f.endswith(".tflite"):
                    print(f"Your TFLite model is here: {os.path.join(output_folder, f)}")
        print("=" * 40)
    else:
        print("❌ Conversion failed. Please check the logs above.")


if __name__ == "__main__":
    export()