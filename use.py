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
from torch.nn import init

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

class MLP(nn.Module):
    def __init__(self, dim, dropout):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
        )
        self._initialize_weights_improved()

    def _initialize_weights_improved(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.Embedding):
                init.normal_(m.weight, mean=0, std=0.02)
            elif isinstance(m, nn.LayerNorm):
                init.constant_(m.weight, 1.0)
                init.constant_(m.bias, 0.0)
    def forward(self, x):
        return self.mlp(x)

class CovarianceColliderLayer(nn.Module):
    def __init__(self, dim, features_dim, heads, covariance_cnt, second_projection_dim, dropout):
        super().__init__()
        self.dim = dim
        self.features_dim = features_dim
        self.heads = heads
        self.head_dim = dim // heads
        self.covariance_cnt = covariance_cnt
        self.low_rank = 64
        self.db_rank = 32
        self.Q_layer = nn.Linear(dim, dim, bias=False)
        self.K_layer = nn.Linear(dim, dim, bias=False)
        self.V_layer = nn.Linear(dim, dim, bias=False)
        self.main_q_to_db = nn.Linear(features_dim, features_dim, bias=False)
        self.main_k_to_db = nn.Linear(features_dim, features_dim, bias=False)
        self.main_v_to_db = nn.Linear(features_dim, features_dim, bias=False)
        self.main_kernel_q = nn.Linear(features_dim, self.low_rank)
        self.main_kernel_k = nn.Linear(features_dim, self.low_rank)
        self.q_layers = nn.ModuleList(
            [nn.Linear(features_dim, features_dim, bias=False) for _ in range(covariance_cnt)])
        self.k_layers = nn.ModuleList(
            [nn.Linear(features_dim, features_dim, bias=False) for _ in range(covariance_cnt)])
        self.v_layers = nn.ModuleList(
            [nn.Linear(features_dim, features_dim, bias=False) for _ in range(covariance_cnt)])
        self.step_kernel_q = nn.ModuleList([nn.Linear(features_dim, self.low_rank) for _ in range(covariance_cnt)])
        self.step_kernel_k = nn.ModuleList([nn.Linear(features_dim, self.low_rank) for _ in range(covariance_cnt)])
        self.qkv_mlp_layers = nn.ModuleList([MLP(features_dim, dropout) for _ in range(covariance_cnt)])
        self.qkv_norm_layers1 = nn.ModuleList([nn.LayerNorm(features_dim) for _ in range(covariance_cnt)])
        self.qkv_norm_layers2 = nn.ModuleList([nn.LayerNorm(features_dim) for _ in range(covariance_cnt)])
        self.main_data_base_u = nn.Parameter(torch.randn(heads, features_dim, self.db_rank))
        self.main_data_base_v = nn.Parameter(torch.randn(heads, self.db_rank, features_dim))
        self.collider_db_u = nn.ParameterList(
            [nn.Parameter(torch.randn(heads, features_dim, self.db_rank)) for _ in range(covariance_cnt)])
        self.collider_db_v = nn.ParameterList(
            [nn.Parameter(torch.randn(heads, self.db_rank, features_dim)) for _ in range(covariance_cnt)])
        self.second_projection_layers1 = nn.ParameterList(
            [nn.Parameter(torch.randn(heads, features_dim, second_projection_dim)) for _ in range(covariance_cnt)])
        self.second_projection_layers2 = nn.ParameterList(
            [nn.Parameter(torch.randn(heads, features_dim, second_projection_dim)) for _ in range(covariance_cnt)])
        for p in [self.main_data_base_u, self.main_data_base_v] + list(self.collider_db_u) + list(
                self.collider_db_v) + list(self.second_projection_layers1) + list(self.second_projection_layers2):
            init.orthogonal_(p)
        self.proj_matrix = nn.Parameter(torch.randn(heads, self.head_dim, features_dim))
        self.proj_matrix1 = nn.Parameter(torch.randn(heads, self.head_dim, features_dim))
        self.proj_matrix2 = nn.Parameter(torch.randn(heads, self.head_dim, features_dim))
        self.proj_matrix3 = nn.Parameter(torch.randn(heads, features_dim, self.head_dim))
        init.orthogonal_(self.proj_matrix)
        init.orthogonal_(self.proj_matrix1)
        init.orthogonal_(self.proj_matrix2)
        init.orthogonal_(self.proj_matrix3)
        self.mlp1 = MLP(dim, dropout)
        self.main_mlp_to_db = MLP(features_dim, dropout)
        self.final_mlp = MLP(dim, dropout)
        self.main_norm_to_db1 = nn.LayerNorm(features_dim)
        self.main_norm_to_db2 = nn.LayerNorm(features_dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.covariance_cnt_layer1 = nn.Linear(self.head_dim, covariance_cnt)
        self.covariance_cnt_layer2 = nn.Linear(covariance_cnt, covariance_cnt)
        self._initialize_weights_improved()

    def _initialize_weights_improved(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b, s, d = x.shape
        h = self.heads
        x = self.mlp1(x)
        q = self.Q_layer(x).view(b, s, h, self.head_dim).transpose(1, 2)
        k = self.K_layer(x).view(b, s, h, self.head_dim).transpose(1, 2)
        v = self.V_layer(x).view(b, s, h, self.head_dim).transpose(1, 2)
        q_proj = torch.einsum("bhsd,hdf->bhsf", q, self.proj_matrix)
        q_proj = F.elu(q_proj) + 1
        cv = torch.matmul(k.transpose(-1, -2), v)
        cv_proj = torch.einsum("bhdd,hdf->bhdf", cv, self.proj_matrix1)
        cv_proj = F.elu(cv_proj) + 1
        descriptor = torch.matmul(cv_proj.transpose(-1, -2), self.proj_matrix2.unsqueeze(0).expand(b, -1, -1, -1))
        descriptor = F.elu(descriptor) + 1
        main_db = torch.matmul(self.main_data_base_u, self.main_data_base_v).unsqueeze(0).expand(b, -1, -1, -1)
        q_main = self.main_q_to_db(descriptor)
        k_main = self.main_k_to_db(main_db)
        v_main = self.main_v_to_db(main_db)
        phi_q = F.elu(self.main_kernel_q(q_main)) + 1
        phi_k = F.elu(self.main_kernel_k(k_main)) + 1
        kv = torch.matmul(phi_k.transpose(-2, -1), v_main)
        out_main = torch.matmul(phi_q, kv)
        denom = torch.matmul(phi_q.sum(dim=-2, keepdim=True), phi_k.sum(dim=-2, keepdim=True).transpose(-2, -1)) + 1e-8
        out_main = out_main / denom
        out_main = descriptor + out_main
        out_main = self.main_norm_to_db2(out_main + self.main_mlp_to_db(self.main_norm_to_db1(out_main)))
        depth = F.softmax(self.covariance_cnt_layer2(F.elu(self.covariance_cnt_layer1(q).mean(-2)) + 1), dim=-1)
        collider_out = out_main.clone()
        for i in range(self.covariance_cnt):
            p1 = F.elu(torch.matmul(collider_out, self.second_projection_layers1[i])) + 1
            p2 = F.elu(torch.matmul(collider_out, self.second_projection_layers2[i])) + 1
            step_out = torch.matmul(p1, p2.transpose(-1, -2))
            db = torch.matmul(self.collider_db_u[i], self.collider_db_v[i]).unsqueeze(0).expand(b, -1, -1, -1)
            q_step = self.q_layers[i](step_out)
            k_step = self.k_layers[i](db)
            v_step = self.v_layers[i](db)
            phi_q_s = F.elu(self.step_kernel_q[i](q_step)) + 1
            phi_k_s = F.elu(self.step_kernel_k[i](k_step)) + 1
            kv_s = torch.matmul(phi_k_s.transpose(-2, -1), v_step)
            out_step = torch.matmul(phi_q_s, kv_s)
            denom_s = torch.matmul(phi_q_s.sum(dim=-2, keepdim=True),
                                   phi_k_s.sum(dim=-2, keepdim=True).transpose(-2, -1)) + 1e-8
            out_step = out_step / denom_s
            out_step = q_step + out_step
            out_step = self.qkv_norm_layers2[i](
                q_step + out_step + self.qkv_mlp_layers[i](self.qkv_norm_layers1[i](out_step)))
            collider_out = collider_out + depth[..., i].unsqueeze(-1).unsqueeze(-1) * out_step
        final_db = F.elu(torch.einsum("bhff,hfd->bhfd", collider_out, self.proj_matrix3)) + 1
        attn_out = torch.einsum("bhsf,bhfd->bhsd", q_proj, final_db).transpose(1, 2).reshape(b, s, -1)
        x = x + attn_out
        x = self.norm2(x + self.final_mlp(self.norm1(x)))
        return x

class ImageClassificationModel(nn.Module):
    def __init__(self, image_size=40, patch_size=4, num_classes=6, dim=256, depth=4, heads=8, features_dim=128,
                 covariance_cnt=4, second_projection_dim=64, dropout=0.1):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.dim = dim
        self.patch_embed = nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches + 1, dim))
        self.layers = nn.ModuleList([
            CovarianceColliderLayer(dim, features_dim, heads, covariance_cnt, second_projection_dim, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)
        self._initialize_weights()

    def _initialize_weights(self):
        init.normal_(self.cls_token, std=0.02)
        init.normal_(self.pos_embed, std=0.02)
        init.xavier_uniform_(self.head.weight)
        init.constant_(self.head.bias, 0)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return self.head(x[:, 0])

class ObjectDetector:
    def __init__(self, heatmap_model_path='best_unet_heatmap.pth', classifier_model_path='final_best_model.pth'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        self.heatmap_model = UNetHeatmap(n_channels=3, n_classes=1).to(self.device)
        self.heatmap_model.load_state_dict(torch.load(heatmap_model_path, map_location=self.device, weights_only=True))
        self.heatmap_model.eval()
        self.classifier_model = ImageClassificationModel(
            image_size=40,
            patch_size=4,
            num_classes=6,
            dim=192,
            depth=3,
            heads=6,
            features_dim=96,
            covariance_cnt=3,
            second_projection_dim=48,
            dropout=0.1
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
        classifier_model_path='img_cognition_final3.pth'
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