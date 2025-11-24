import os
import torch.utils.data as data
import torch
from PIL import Image, ImageFilter
import numpy as np


class TrainDataset(data.Dataset):
    def __init__(self, path_to_imgs, class_to_index, transform=None, padding=5, blur_radius=2, threshold=100):
        self.transform = transform
        self.class_to_index = class_to_index
        self.padding = padding
        self.blur_radius = blur_radius
        self.threshold = threshold

        self.samples = []
        for filedir in os.listdir(path_to_imgs):
            for filename in os.listdir(os.path.join(path_to_imgs, filedir)):
                if filename.endswith(('.jpg', )):
                    base_name = os.path.splitext(filename)[0]
                    img_path = os.path.join(path_to_imgs, filedir, filename)
                    txt_path = os.path.join(path_to_imgs, filedir, base_name + "_text.txt")
                    self.samples.append((img_path, txt_path))

    def __len__(self):
        return len(self.samples)

    def _crop_object(self, image):
        # Размытие для уменьшения шумов
        blurred = image.filter(ImageFilter.GaussianBlur(self.blur_radius))
        blurred_array = np.array(blurred)

        mask = np.all(blurred_array > self.threshold, axis=-1)
        coords = np.nonzero(~mask)

        if len(coords[0]) == 0:
            return image

        # Границы объекта с отступами
        top, bottom = np.min(coords[0]), np.max(coords[0])
        left, right = np.min(coords[1]), np.max(coords[1])

        left = max(0, left - self.padding)
        top = max(0, top - self.padding)
        right = min(image.width, right + self.padding)
        bottom = min(image.height, bottom + self.padding)

        # Обрезка изображения
        cropped = image.crop((left, top, right, bottom))
        return cropped

    def __getitem__(self, index):
        img_path, label_path = self.samples[index]
        image = Image.open(img_path).convert("RGB")

        image = self._crop_object(image)

        if self.transform:
            image = self.transform(image)

        with open(label_path, "r", encoding='utf-8') as f:
            label = float(f.read().strip())

        return image, torch.tensor(label, dtype=torch.long)

    def get_num_classes(self):
        return len(self.class_to_index)  # Исправлено: возвращаем количество классов