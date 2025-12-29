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
        # input is CHW
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


if __name__ == '__main__':
    device = "mps"
    model = UNetHeatmap(n_channels=3, n_classes=1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.BCELoss()

    train_loader = DataLoader(HeatmapDataset(root_dir='датасет 2', sigma=15, target_size=(256, 256)),
                              batch_size=4, shuffle=True, num_workers=4)

    num_epochs = 70
    best_train_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        start_time = time.time()

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        epoch_time = time.time() - start_time

        print(f'Epoch {epoch + 1}/{num_epochs} | '
              f'Train Loss: {train_loss:.6f} | '
              f'Time: {epoch_time:.2f}s')

        if train_loss < best_train_loss:
            best_train_loss = train_loss
            torch.save(model.state_dict(), 'best_unet_heatmap1.pth')