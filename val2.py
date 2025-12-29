import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from PIL import Image
from torchvision import transforms
import cv2
import numpy as np
import os
import random


def process_image(image_path, output_folder):
    img = cv2.imread(image_path)
    if img is None:
        return None

    os.makedirs(output_folder, exist_ok=True)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    mean_a = np.mean(a_channel)
    mean_b = np.mean(b_channel)

    color_distance = np.sqrt((a_channel - mean_a) ** 2 + (b_channel - mean_b) ** 2)

    _, mask = cv2.threshold(color_distance.astype(np.uint8), 15, 255, cv2.THRESH_BINARY)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_contour_area = 50
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]

    os.makedirs(output_folder, exist_ok=True)

    for i, cnt in enumerate(filtered_contours):
        M = cv2.moments(cnt)
        if M['m00'] == 0:
            continue

        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])

        x1 = max(cx - 70, 0)
        y1 = max(cy - 70, 0)
        x2 = min(cx + 70, img.shape[1])
        y2 = min(cy + 70, img.shape[0])

        cropped = img[y1:y2, x1:x2]

        if cropped.size == 0:
            continue

        resized = cv2.resize(cropped, (40, 40), Image.BILINEAR)

        path = f'{output_folder}/part_{random.randint(1, 1000000)}.jpg'
        cv2.imwrite(path, resized)

        return path

    return None


# Новая модель классификации
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
    def __init__(self, dim=192, heads=8, feature_dim=96, dropout=0.1):
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


class MLPBlock(nn.Module):
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
    def __init__(self, embed_dim=192, num_heads=8, mlp_ratio=4, dropout=0.1, feature_dim=96):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = LinearPerformerAttention(embed_dim, num_heads, feature_dim, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = MLPBlock(embed_dim, mlp_hidden_dim, embed_dim, dropout)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ViT(nn.Module):
    def __init__(self, img_size=40, patch_size=4, in_channels=3, num_classes=6, embed_dim=192, depth=10, num_heads=8,
                 feature_dim=96):
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


class ImageClassifier:
    def __init__(self, model_path, class_names=None):
        """Инициализация классификатора с загруженной моделью"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Параметры модели (должны совпадать с обученной моделью)
        self.model_params = {
            'img_size': 40,
            'patch_size': 4,
            'num_classes': 6,
            'embed_dim': 192,
            'depth': 10,
            'num_heads': 8,
            'feature_dim': 96
        }

        # Имена классов (можно изменить под свои нужды)
        self.class_names = class_names or [
            "Class 0", "Class 1", "Class 2",
            "Class 3", "Class 4", "Class 5"
        ]

        # Трансформации для изображения (такие же как при валидации)
        self.transform = transforms.Compose([
            transforms.Resize((40, 40)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # Загрузка модели
        self.model = self._load_model(model_path)

    def _load_model(self, model_path):
        """Загрузка обученной модели"""
        model = ViT(**self.model_params)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to(self.device)
        model.eval()  # Переводим модель в режим инференса
        return model

    def preprocess_image(self, image_path):
        """Предобработка изображения"""
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image)
        image_tensor = image_tensor.unsqueeze(0)  # Добавляем batch dimension
        return image_tensor.to(self.device)

    def predict(self, image_path, top_k=3):
        """Предсказание класса для изображения"""
        # Предобработка
        image_tensor = self.preprocess_image(image_path)

        # Предсказание
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = F.softmax(outputs, dim=1)

        # Получаем топ-K предсказаний
        probs, indices = torch.topk(probabilities, top_k)

        # Преобразуем в удобный формат
        results = []
        for i in range(top_k):
            class_idx = indices[0, i].item()
            confidence = probs[0, i].item()
            class_name = self.class_names[class_idx] if class_idx < len(self.class_names) else f"Class {class_idx}"
            results.append({
                'class': class_idx,
                'class_name': class_name,
                'confidence': confidence * 100
            })

        return results

    def predict_batch(self, image_paths, top_k=3):
        """Предсказание для нескольких изображений"""
        batch_tensors = []

        for img_path in image_paths:
            image = Image.open(img_path).convert('RGB')
            image_tensor = self.transform(image)
            batch_tensors.append(image_tensor)

        batch_tensor = torch.stack(batch_tensors).to(self.device)

        with torch.no_grad():
            outputs = self.model(batch_tensor)
            probabilities = F.softmax(outputs, dim=1)

        all_results = []
        for i, img_path in enumerate(image_paths):
            probs, indices = torch.topk(probabilities[i], top_k)
            results = []
            for j in range(top_k):
                class_idx = indices[j].item()
                confidence = probs[j].item()
                class_name = self.class_names[class_idx] if class_idx < len(self.class_names) else f"Class {class_idx}"
                results.append({
                    'class': class_idx,
                    'class_name': class_name,
                    'confidence': confidence * 100
                })
            all_results.append({
                'image_path': img_path,
                'predictions': results
            })

        return all_results


def main_inference():
    model_path = "model_weights_last4.pth"

    class_names = [
        "Class 0", "Class 1", "Class 2",
        "Class 3", "Class 4", "Class 5"
    ]

    classifier = ImageClassifier(model_path, class_names)

    ar = os.listdir("датасет 3")
    accuracy_ = 0
    m_cnt = 0
    for folder in ar:
        img_ar = os.listdir("датасет 3/" + folder)
        cnt = 0
        max_el_in_folder = 0
        for img in img_ar:
            path = f"датасет 3/{folder}/{img}"
            path = process_image(path, "output_parts")
            if path is not None:
                max_el_in_folder += 1
                results = classifier.predict(path, top_k=3)
                max_confidence = 0
                out_class = None
                for result in results:
                    confidence = result["confidence"]
                    if confidence > max_confidence:
                        max_confidence = confidence
                        out_class = result["class"]
                if out_class == int(folder):
                    cnt += 1
        if max_el_in_folder:
            m_cnt += 1
            local_accuracy = cnt / max_el_in_folder
            accuracy_ += local_accuracy
    print(f"Overall accuracy: {accuracy_ / m_cnt:.4f}")


if __name__ == "__main__":
    main_inference()