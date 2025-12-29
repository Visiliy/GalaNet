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
import torch.nn.init as init

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

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=40, patch_size=4, in_channels=3, embed_dim=192):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

class LinearPerformerAttention(nn.Module):
    def __init__(self, dim=192, heads=6, feature_dim=96, dropout=0.1):
        super().__init__()
        self.heads = heads
        self.feature_dim = feature_dim
        self.head_dim = dim // heads
        self.proj_matrix = nn.Parameter(torch.randn(heads, self.head_dim, feature_dim))
        nn.init.orthogonal_(self.proj_matrix)
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    init.zeros_(m.bias)

    def forward(self, x, mask=None):
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(-1)
            x = x * mask.to(x.dtype)
        b, n, d = x.shape
        h = self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.reshape(b, n, h, -1).transpose(1, 2), qkv)
        q_proj = torch.einsum('bhnd,hdf->bhnf', q, self.proj_matrix)
        k_proj = torch.einsum('bhnd,hdf->bhnf', k, self.proj_matrix)
        q_proj = F.elu(q_proj) + 1
        k_proj = F.elu(k_proj) + 1
        k_v = torch.einsum('bhnf,bhnd->bhfd', k_proj, v)
        attention_out = torch.einsum('bhnf,bhfd->bhnd', q_proj, k_v)
        k_proj_sum = k_proj.sum(dim=2, keepdim=True)
        z = 1.0 / (torch.einsum('bhnf,bhf->bhn', q_proj, k_proj_sum.squeeze(2)) + 1e-8)
        attention_out = attention_out * z.unsqueeze(-1)
        attention_out = attention_out.transpose(1, 2).reshape(b, n, -1)
        out = self.to_out(attention_out)
        return out

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim=192, num_heads=6, mlp_ratio=4, dropout=0.1, feature_dim=96):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = LinearPerformerAttention(embed_dim, num_heads, feature_dim, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = MLP(embed_dim, mlp_hidden_dim, embed_dim, dropout)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class ViT(nn.Module):
    def __init__(self, img_size=40, patch_size=4, in_channels=3, num_classes=6, embed_dim=192, depth=3, num_heads=6, feature_dim=96):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.patch_embed.num_patches, embed_dim))
        self.pos_drop = nn.Dropout(0.1)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, feature_dim=feature_dim) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        x = self.head(x[:, 0])
        return x

class ObjectDetector:
    def __init__(self, heatmap_model_path='best_unet_heatmap.pth', classifier_model_path='final_best_model.pth'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        self.heatmap_model = UNetHeatmap(n_channels=3, n_classes=1).to(self.device)
        self.heatmap_model.load_state_dict(torch.load(heatmap_model_path, map_location=self.device, weights_only=True))
        self.heatmap_model.eval()
        self.classifier_model = ViT(
            img_size=40,
            patch_size=4,
            embed_dim=192,
            depth=10,
            num_heads=8,
            num_classes=6,
            feature_dim=96
        ).to(self.device)
        self.classifier_model.load_state_dict(torch.load(classifier_model_path, map_location=self.device, weights_only=True))
        self.classifier_model.eval()
        total_params = sum(p.numel() for p in self.classifier_model.parameters())
        print(f"Всего параметров: {total_params}")
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.classifier_transform = T.Compose([
            T.Resize((40, 40)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def detect_objects(self, image_path, heatmap_threshold=0.7):
        image = Image.open(image_path).convert('RGB')
        orig_w, orig_h = image.size
        tensor = self.transform(image.resize((256, 256), Image.BILINEAR)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            pred = self.heatmap_model(tensor).squeeze().cpu().numpy()
        pred = np.where(pred >= heatmap_threshold, 1.0, 0.0)
        pred_img = Image.fromarray((pred * 255).astype(np.uint8)).resize((orig_w, orig_h), Image.NEAREST)
        binary_map = np.array(pred_img) > 0
        labeled, num_features = label(binary_map)
        centers = center_of_mass(binary_map, labeled, range(1, num_features + 1))
        centers = [(int(cy), int(cx)) for cy, cx in centers if not (np.isnan(cy) or np.isnan(cx))]
        detections = []
        for cy, cx in centers:
            crop_size = 140
            half_size = crop_size // 2
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
                crop = padded_image.crop((new_left, new_top, new_right, new_bottom))
            else:
                crop = image.crop((left, top, right, bottom))
            if crop.size[0] < 40 or crop.size[1] < 40:
                continue
            crop = crop.resize((40, 40), Image.BILINEAR)
            detections.append({
                'center': (cx, cy),
                'crop': crop
            })
        return detections, image

    def classify_objects(self, detections):
        results = []
        for det in detections:
            crop_tensor = self.classifier_transform(det['crop']).unsqueeze(0).to(self.device)
            with torch.no_grad():
                output = self.classifier_model(crop_tensor)
                probabilities = F.softmax(output, dim=1)
                confidence, predicted_class = torch.max(probabilities, 1)
            results.append({
                'center': det['center'],
                'class': predicted_class.item(),
                'confidence': confidence.item()
            })
        return results

    def process_image(self, image_path, heatmap_threshold=0.7):
        detections, original_image = self.detect_objects(image_path, heatmap_threshold)
        results = self.classify_objects(detections)
        return {
            'image_path': image_path,
            'original_size': original_image.size,
            'detections': results
        }

def main():
    detector = ObjectDetector(
        heatmap_model_path='best_unet_heatmap.pth',
        classifier_model_path='model_weights_last4.pth'
    )
    image_path = "WIN_20251219_16_22_41_Pro.jpg"
    if not os.path.exists(image_path):
        print("Файл не найден!")
        return
    print(f"Обработка изображения: {image_path}")
    result = detector.process_image(image_path)
    print(f"\nРезультаты для {result['image_path']}:")
    print(f"Размер изображения: {result['original_size']}")
    print(f"Найдено объектов: {len(result['detections'])}")
    for i, det in enumerate(result['detections']):
        print(f"\nОбъект {i+1}:")
        print(f"  Координаты центра: {det['center']}")
        print(f"  Класс: {det['class']}")
        print(f"  Уверенность: {det['confidence']:.4f}")

if __name__ == '__main__':
    main()