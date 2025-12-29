import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import glob
from scipy.ndimage import label, center_of_mass


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
            T.RandomHorizontalFlip(),
            T.RandomRotation(10),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
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


class InferenceDataset(Dataset):
    def __init__(self, root_dir, target_size=(256, 256)):
        self.root_dir = root_dir
        self.target_size = target_size
        self.samples = []

        for img_path in sorted(glob.glob(os.path.join(root_dir, '**', '*.*'), recursive=True)):
            if img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                self.samples.append(img_path)

        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        orig_size = image.size
        image_resized = image.resize(self.target_size, Image.BILINEAR)
        tensor = self.transform(image_resized)
        return tensor, img_path, orig_size


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNetHeatmap(nn.Module):
    def __init__(self, n_channels=3, n_classes=1, bilinear=True):
        super(UNetHeatmap, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return torch.sigmoid(logits)


class HeatmapDetector:
    def __init__(self, model_path, device='mps'):
        self.device = torch.device(device if torch.backends.mps.is_available() else 'cpu')
        self.model = UNetHeatmap(n_channels=3, n_classes=1).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.threshold = 0.7  # Порог бинаризации для тепловой карты
        self.crop_size = 140  # Размер кропа (140x140 пикселей)

    def preprocess_image(self, image_path, target_size=(256, 256)):
        image = Image.open(image_path).convert('RGB')
        orig_size = image.size
        image_resized = image.resize(target_size, Image.BILINEAR)
        image_tensor = self.transform(image_resized).unsqueeze(0)
        return image_tensor, orig_size, image

    def detect_and_crop_objects(self, image_path, output_dir=None, save_crops=True):
        image_tensor, orig_size, original_image = self.preprocess_image(image_path)
        orig_w, orig_h = orig_size

        with torch.no_grad():
            heatmap = self.model(image_tensor.to(self.device))

        # Преобразуем тепловую карту в бинарную маску
        heatmap_np = heatmap.squeeze().cpu().numpy()
        binary_mask = np.where(heatmap_np >= self.threshold, 1.0, 0.0)

        # Масштабируем бинарную маску до оригинального размера
        binary_img = Image.fromarray((binary_mask * 255).astype(np.uint8))
        binary_img_resized = binary_img.resize((orig_w, orig_h), Image.NEAREST)
        binary_map = np.array(binary_img_resized) > 0

        # Находим связанные компоненты и их центры масс
        labeled, num_features = label(binary_map)
        centers = center_of_mass(binary_map, labeled, range(1, num_features + 1))

        # Фильтруем некорректные центры и преобразуем координаты
        filtered_centers = []
        for cy, cx in centers:
            if not (np.isnan(cy) or np.isnan(cx)):
                cy_int = int(cy)
                cx_int = int(cx)
                filtered_centers.append((cy_int, cx_int))

        centers = filtered_centers

        crops = []
        half_size = self.crop_size // 2

        for i, (cy, cx) in enumerate(centers):
            # Вычисляем границы кропа
            left = cx - half_size
            top = cy - half_size
            right = cx + half_size
            bottom = cy + half_size

            # Проверяем, выходит ли кроп за границы изображения
            pad_left = max(0, -left)
            pad_top = max(0, -top)
            pad_right = max(0, right - orig_w)
            pad_bottom = max(0, bottom - orig_h)

            if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
                # Создаем изображение с padding
                new_w = orig_w + pad_left + pad_right
                new_h = orig_h + pad_top + pad_bottom
                padded_image = Image.new('RGB', (new_w, new_h), (0, 0, 0))
                padded_image.paste(original_image, (pad_left, pad_top))

                # Корректируем координаты центра
                new_center_x = cx + pad_left
                new_center_y = cy + pad_top

                # Вычисляем новые границы кропа
                new_left = new_center_x - half_size
                new_top = new_center_y - half_size
                new_right = new_center_x + half_size
                new_bottom = new_center_y + half_size

                # Выполняем кроп
                crop = padded_image.crop((new_left, new_top, new_right, new_bottom))
            else:
                # Если кроп полностью внутри изображения
                crop = original_image.crop((left, top, right, bottom))

            # Масштабируем до 40x40 пикселей
            crop_resized = crop.resize((40, 40), Image.BILINEAR)
            crops.append(crop_resized)

            # Сохраняем кропы, если нужно
            if save_crops and output_dir:
                base_name = os.path.splitext(os.path.basename(image_path))[0]
                crop_filename = os.path.join(output_dir, f"{base_name}_obj{i:03d}.png")
                crop_resized.save(crop_filename)

        result = {
            'count': len(centers),
            'coordinates': [(cx, cy) for cy, cx in centers],  # Возвращаем (x, y)
            'heatmap': heatmap_np,
            'binary_mask': binary_mask,
            'crops': crops
        }

        return result

    def predict_single_image(self, image_path):
        """Простой предсказатель для одной картинки"""
        result = self.detect_and_crop_objects(image_path, output_dir=None, save_crops=False)

        print(f"\nКоличество деталей: {result['count']}")
        if result['count'] > 0:
            print(f"Координаты центров деталей (x, y):")
            for i, (x, y) in enumerate(result['coordinates'], 1):
                print(f"  Деталь {i}: ({x}, {y})")
        else:
            print("Детали не обнаружены.")

        return result

    def process_folder(self, folder_path, output_dir='cropped_objects'):
        """Обработка всех изображений в папке"""
        os.makedirs(output_dir, exist_ok=True)

        inference_dataset = InferenceDataset(root_dir=folder_path, target_size=(256, 256))

        for idx, (tensor, img_path, orig_size) in enumerate(inference_dataset):
            print(f"\nОбработка: {img_path}")

            tensor = tensor.unsqueeze(0).to(self.device)

            with torch.no_grad():
                heatmap = self.model(tensor)

            heatmap_np = heatmap.squeeze().cpu().numpy()
            binary_mask = np.where(heatmap_np >= self.threshold, 1.0, 0.0)

            # Масштабируем до оригинального размера
            orig_w, orig_h = orig_size
            binary_img = Image.fromarray((binary_mask * 255).astype(np.uint8))
            binary_img_resized = binary_img.resize((orig_w, orig_h), Image.NEAREST)
            binary_map = np.array(binary_img_resized) > 0

            # Находим центры
            labeled, num_features = label(binary_map)
            centers = center_of_mass(binary_map, labeled, range(1, num_features + 1))
            centers = [(int(cy), int(cx)) for cy, cx in centers if not (np.isnan(cy) or np.isnan(cx))]

            original_image = Image.open(img_path).convert('RGB')

            # Сохраняем кропы
            for i, (cy, cx) in enumerate(centers):
                crop_result = self._crop_around_center(original_image, cx, cy, orig_w, orig_h)
                if crop_result:
                    crop_resized = crop_result.resize((40, 40), Image.BILINEAR)
                    base_name = os.path.splitext(os.path.basename(img_path))[0]
                    crop_filename = os.path.join(output_dir, f"{base_name}_obj{i:03d}.png")
                    crop_resized.save(crop_filename)

            print(f"Найдено объектов: {len(centers)}")

    def _crop_around_center(self, image, cx, cy, orig_w, orig_h):
        """Вспомогательная функция для кропа вокруг центра"""
        half_size = self.crop_size // 2

        left = cx - half_size
        top = cy - half_size
        right = cx + half_size
        bottom = cy + half_size

        pad_left = max(0, -left)
        pad_top = max(0, -top)
        pad_right = max(0, right - orig_w)
        pad_bottom = max(0, bottom - orig_h)

        if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
            new_w = orig_w + pad_left + pad_right
            new_h = orig_h + pad_top + pad_bottom
            padded_image = Image.new('RGB', (new_w, new_h), (0, 0, 0))
            padded_image.paste(image, (pad_left, pad_top))

            new_center_x = cx + pad_left
            new_center_y = cy + pad_top

            new_left = new_center_x - half_size
            new_top = new_center_y - half_size
            new_right = new_center_x + half_size
            new_bottom = new_center_y + half_size

            return padded_image.crop((new_left, new_top, new_right, new_bottom))
        else:
            return image.crop((left, top, right, bottom))


if __name__ == '__main__':
    # Инициализация детектора
    detector = HeatmapDetector('best_unet_heatmap1.pth')

    # Пример 1: Обработка одного изображения
    image_path = "WIN_20251219_16_22_41_Pro.jpg"

    if os.path.exists(image_path):
        result = detector.predict_single_image(image_path)

        # Сохранение тепловой карты для визуализации
        heatmap_img = Image.fromarray((result['heatmap'] * 255).astype(np.uint8))
        heatmap_img.save('heatmap_visualization.png')
        print(f"\nТепловая карта сохранена как 'heatmap_visualization.png'")
    else:
        print(f"Файл {image_path} не найден.")

    # Пример 2: Обработка всей папки с сохранением кропов
    # detector.process_folder('датасет 2', output_dir='cropped_objects')