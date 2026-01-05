import torch.utils.data as data
from PIL import Image
import glob
import os
import torchvision.transforms as transforms

class LowLightDataset(data.Dataset):
    def __init__(self, dataset_dir):
        # 路径指向: dataset/LOLdataset/our485/low
        self.image_list = glob.glob(os.path.join(dataset_dir, 'our485', 'low', "*.png"))

        if len(self.image_list) == 0:
            print(f"Error: No images found in {dataset_dir}. Check path structure!")
            print(f"Expected path: {os.path.join(dataset_dir, 'our485', 'low')}")

        self.transform = transforms.Compose([
            # === 核心修改: 提升分辨率 ===
            # 从 256 改为 512。这对于让 TFLite 模型在手机上输出清晰大图至关重要。
            transforms.Resize((512, 512)),
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        img_path = self.image_list[index]
        data = Image.open(img_path).convert('RGB')
        data = self.transform(data)
        return data

    def __len__(self):
        return len(self.image_list)