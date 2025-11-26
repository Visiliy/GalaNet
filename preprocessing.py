import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import Dataset
import glob
from scipy.ndimage import label, center_of_mass



class HeatmapDataset(Dataset):
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


device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

model = UNetHeatmap(n_channels=3, n_classes=1).to(device)
model.load_state_dict(torch.load('best_unet_heatmap.pth', map_location=device, weights_only=True))
model.eval()

dataset = HeatmapDataset(root_dir='датасет 2', target_size=(256, 256))
output_dir = 'cropped_objects'
os.makedirs(output_dir, exist_ok=True)

with torch.no_grad():
    for idx, (tensor, img_path, orig_size) in enumerate(dataset):
        orig_w, orig_h = orig_size
        tensor = tensor.unsqueeze(0).to(device)
        pred = model(tensor).squeeze().cpu().numpy()

        pred = np.where(pred >= 0.7, 1.0, 0.0)
        pred_img = Image.fromarray((pred * 255).astype(np.uint8)).resize((orig_w, orig_h), Image.NEAREST)
        binary_map = np.array(pred_img) > 0

        labeled, num_features = label(binary_map)
        centers = center_of_mass(binary_map, labeled, range(1, num_features + 1))
        centers = [(int(cy), int(cx)) for cy, cx in centers if not (np.isnan(cy) or np.isnan(cx))]

        original_image = Image.open(img_path).convert('RGB')

        for i, (cy, cx) in enumerate(centers):
            left = max(0, cx - 50)
            top = max(0, cy - 50)
            right = min(orig_w, cx + 50)
            bottom = min(orig_h, cy + 50)

            if right - left < 40 or bottom - top < 40:
                continue

            crop = original_image.crop((left, top, right, bottom))
            if crop.size != (40, 40):
                crop = crop.resize((40, 40), Image.BILINEAR)

            base_name = os.path.splitext(os.path.basename(img_path))[0]
            crop.save(os.path.join(output_dir, f"{base_name}_obj{i:03d}.png"))

        print(f"Processed {img_path}: found {len(centers)} objects")