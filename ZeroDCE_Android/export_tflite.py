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
        # 只需要增强后的图，不需要参数图
        enhanced, _ = self.model(x)
        return enhanced

def export():
    weights_path = "weights/ZeroDCE_final.pth"
    onnx_path = "weights/ZeroDCE_dynamic.onnx" # 改名，标记为动态
    output_folder = "weights/tflite_output"

    if not os.path.exists(weights_path):
        print(f"Error: {weights_path} not found. Run train.py first!")
        return

    print(">>> Loading PyTorch model...")
    model = C_DCE_Net()
    try:
        model.load_state_dict(torch.load(weights_path, map_location='cpu'))
    except:
        print("Warning: Loading weights strictly failed, trying strict=False...")
        model.load_state_dict(torch.load(weights_path, map_location='cpu'), strict=False)

    model.eval()
    deploy_model = DeployModel(model)

    # --- 关键修改点 1: 定义动态轴 ---
    # 这告诉 ONNX，高度(dim 2)和宽度(dim 3)是可变的
    dynamic_axes = {
        'input': {2: 'height', 3: 'width'},
        'output': {2: 'height', 3: 'width'}
    }

    # 哪怕是动态模型，也需要一个 Dummy Input 来跑通一次图
    # 这里我们用一个常见的高清比例，比如 720p，但这不影响最终模型的灵活性
    dummy_input = torch.randn(1, 3, 640, 360)

    print(">>> Exporting to ONNX with Dynamic Shapes...")
    torch.onnx.export(
        deploy_model,
        dummy_input,
        onnx_path,
        verbose=False,
        input_names=['input'],
        output_names=['output'],
        opset_version=11,
        dynamic_axes=dynamic_axes  # 传入动态轴配置
    )
    print(f"ONNX exported to {onnx_path}")

    # 简化 ONNX
    print(">>> Simplifying ONNX model...")
    try:
        import onnxsim
        import onnx
        onnx_model = onnx.load(onnx_path)
        # 简化时保留动态特性
        model_simp, check = onnxsim.simplify(onnx_model)
        onnx.save(model_simp, onnx_path)
        print("ONNX simplified successfully.")
    except Exception as e:
        print(f"Warning: onnxsim failed ({e}), using original ONNX...")

    print(">>> Converting to TFLite using onnx2tf...")
    # --- 关键修改点 2: onnx2tf 命令 ---
    # onnx2tf 会自动识别 ONNX 中的动态轴。
    # 如果你在安卓上遇到 GPU Delegate 不支持动态 Shape 的问题，
    # 可以将下面的 -ois 移除，或者指定一个较大的固定尺寸（见下文说明）。
    cmd = f"onnx2tf -i {onnx_path} -o {output_folder}"

    process = subprocess.Popen(cmd, shell=True)
    process.wait()

    if process.returncode == 0:
        print("\n" + "=" * 40)
        print("🎉 SUCCESS! Conversion finished.")
        print("注意：在 Android 使用此模型时，需要调用 interpreter.resizeInput() 来适配不同分辨率的图片。")
        print("=" * 40)
    else:
        print("❌ Conversion failed.")

if __name__ == "__main__":
    export()