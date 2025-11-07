import os
import torch.utils.data as data
import torch
from PIL import Image


class TrainDataset(data.Dataset):
    def __init__(self, path_to_imgs, class_to_index, transform=None):
        self.transform = transform

        self.class_to_index = class_to_index

        self.samples = []
        for filedir in os.listdir(path_to_imgs):
            for filename in os.listdir(f"{path_to_imgs}/{filedir}"):
                if filename.endswith(('.jpg', '.png', '.jpeg')):
                    base_name = os.path.splitext(filename)[0]
                    img_path = os.path.join(f"{path_to_imgs}/{filedir}", filename)
                    txt_path = os.path.join(f"{path_to_imgs}/{filedir}", base_name + "_text.txt")
                    self.samples.append((img_path, txt_path))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        img_path, label_path = self.samples[index]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        with open(label_path, "r", encoding='utf-8') as f:
            label= float(f.read().strip("\n"))


        return image, torch.tensor(label, dtype=torch.long)

    def get_num_classes(self):
        return self.class_to_index