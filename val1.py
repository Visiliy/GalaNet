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


class ImageClassifier:
    def __init__(self, model_path, class_names=None):
        """Инициализация классификатора с загруженной моделью"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Параметры модели (должны совпадать с обученной моделью)
        self.model_params = {
            'image_size': 40,
            'patch_size': 4,
            'num_classes': 6,
            'dim': 192,
            'depth': 3,
            'heads': 6,
            'features_dim': 96,
            'covariance_cnt': 3,
            'second_projection_dim': 48,
            'dropout': 0.1
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
        model = ImageClassificationModel(**self.model_params)
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
    model_path = "img_cognition_final3.pth"

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
    print(accuracy_ / m_cnt)




if __name__ == "__main__":
    main_inference()