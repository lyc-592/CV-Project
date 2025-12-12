import torch.utils.data as data
from PIL import Image
import glob
import os
import torchvision.transforms as transforms


class LowLightDataset(data.Dataset):
    def __init__(self, dataset_dir):
        # 路径指向: dataset/LOLdataset/our485/low
        # 注意：这里我们只训练微光增强，所以只需要 low 文件夹的图
        self.image_list = glob.glob(os.path.join(dataset_dir, 'our485', 'low', "*.png"))

        if len(self.image_list) == 0:
            print(f"Error: No images found in {dataset_dir}. Check path structure!")
            print(f"Expected path: {os.path.join(dataset_dir, 'our485', 'low')}")

        self.transform = transforms.Compose([
            # 训练时统一缩放到 256x256，这对于生成 256x256 的 TFLite 模型很重要
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        img_path = self.image_list[index]
        # Image.open 会自动处理 png 格式
        data = Image.open(img_path).convert('RGB')
        data = self.transform(data)
        return data

    def __len__(self):
        return len(self.image_list)