import os
import torch
import torch.utils.data as data
from PIL import Image


class TrainDataset(data.Dataset):
    def __init__(self, path_to_imgs, path_to_labels, classes, transform=None):
        self.transform = transform
        self.classes = classes
        img_files = set(os.path.splitext(f)[0] for f in os.listdir(path_to_imgs))
        label_files = set(os.path.splitext(f)[0] for f in os.listdir(path_to_labels))
        common_ids = sorted(img_files & label_files)
        self.samples = []
        for fid in common_ids:
            img_path = os.path.join(path_to_imgs, fid + ".jpg")
            label_path = os.path.join(path_to_labels, fid + ".txt")
            if not os.path.exists(img_path):
                img_path = os.path.join(path_to_imgs, fid + ".png")

            self.samples.append((img_path, label_path))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        img_path, label_path = self.samples[index]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        with open(label_path, "r") as f:
            label = int(f.read().strip())

        return image, torch.tensor(label, dtype=torch.long)

    def get_num_classes(self):
        return self.classes