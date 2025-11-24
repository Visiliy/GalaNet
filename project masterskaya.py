import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


class HeatmapDataset(Dataset):
    def __init__(self, root_dir, sigma=15, target_size=(256, 256)):
        self.root_dir = root_dir
        self.sigma = sigma
        self.target_size = target_size
        self.samples = []

        for folder_name in sorted(os.listdir(root_dir)):
            folder_path = os.path.join(root_dir, folder_name)
            if os.path.isdir(folder_path):
                for file_name in os.listdir(folder_path):
                    if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        img_path = os.path.join(folder_path, file_name)
                        txt_path = os.path.splitext(img_path)[0] + '.txt'
                        if os.path.exists(txt_path):
                            self.samples.append((img_path, txt_path))

        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.samples)

    def _parse_yolo_obb_annotations(self, path, orig_w, orig_h):
        centers = []
        try:
            with open(path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 9:
                        coords = list(map(float, parts[1:]))
                        x_coords = coords[0::2]
                        y_coords = coords[1::2]
                        cx = sum(x_coords) / 4.0 * orig_w
                        cy = sum(y_coords) / 4.0 * orig_h
                        centers.append((cx, cy))
        except Exception:
            pass
        return centers

    def _create_gaussian_heatmap(self, centers, h, w):
        heatmap = np.zeros((h, w), dtype=np.float32)
        y_grid, x_grid = np.mgrid[0:h, 0:w]
        for cx, cy in centers:
            cx = np.clip(cx, 0, w - 1)
            cy = np.clip(cy, 0, h - 1)
            g = np.exp(-((x_grid - cx) ** 2 + (y_grid - cy) ** 2) / (2 * self.sigma ** 2))
            heatmap = np.maximum(heatmap, g)
        return heatmap

    def __getitem__(self, idx):
        img_path, txt_path = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        orig_w, orig_h = image.size

        centers = self._parse_yolo_obb_annotations(txt_path, orig_w, orig_h)

        heatmap_full = self._create_gaussian_heatmap(centers, orig_h, orig_w)
        heatmap = Image.fromarray(heatmap_full).resize(self.target_size, Image.BILINEAR)
        heatmap = torch.from_numpy(np.array(heatmap)).unsqueeze(0).float()

        image = image.resize(self.target_size, Image.BILINEAR)
        image = self.transform(image)

        return image, heatmap