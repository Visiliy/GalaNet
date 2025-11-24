import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageOps
import numpy as np
import cv2
from torchvision import transforms
import math
import warnings

warnings.filterwarnings('ignore')

try:
    from tensorflow.keras.models import load_model
except ImportError:
    try:
        from keras.models import load_model
    except ImportError:
        load_model = None


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
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_channels, out_channels))

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
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
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
        super().__init__()
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


def select_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def load_models():
    device = select_device()
    unet_model = UNetHeatmap(n_channels=3, n_classes=1).to(device)
    try:
        unet_model.load_state_dict(torch.load('best_unet_heatmap.pth', map_location=device))
    except Exception:
        pass
    unet_model.eval()

    keras_model = None
    if load_model is not None:
        try:
            keras_model = load_model("converted_keras/keras_model.h5", compile=False)
        except Exception:
            keras_model = None

    try:
        with open("converted_keras/labels.txt", "r", encoding="utf-8") as f:
            class_names = [line.strip() for line in f.readlines() if line.strip()]
    except Exception:
        class_names = ["class_0", "class_1"]

    transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    return unet_model, keras_model, class_names, transform, device


def find_components(heatmap, min_area=50):
    heatmap_u8 = np.clip((heatmap * 255).astype(np.uint8), 0, 255)
    blur = cv2.GaussianBlur(heatmap_u8, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    num_labels, labels_im, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity=8)
    centers = []
    boxes = []
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        cx, cy = int(centroids[label][0]), int(centroids[label][1])
        centers.append((cx, cy))
        boxes.append((x, y, w, h, area))
    return centers, boxes, thresh


def clamp_bbox(x, y, w, h, img_w, img_h, pad=30):
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(img_w, x + w + pad)
    y2 = min(img_h, y + h + pad)
    return x1, y1, x2, y2


def non_max_suppression(centers, boxes, iou_thresh=0.3):
    if not centers:
        return []
    rects = []
    for (cx, cy), (x, y, w, h, area) in zip(centers, boxes):
        rects.append([x, y, x + w, y + h, area, cx, cy])
    rects = sorted(rects, key=lambda r: r[4], reverse=True)
    keep = []
    while rects:
        current = rects.pop(0)
        keep.append((int(current[5]), int(current[6]), (current[0], current[1], current[2], current[3])))
        rest = []
        for r in rects:
            xx1 = max(current[0], r[0])
            yy1 = max(current[1], r[1])
            xx2 = min(current[2], r[2])
            yy2 = min(current[3], r[3])
            iw = max(0, xx2 - xx1)
            ih = max(0, yy2 - yy1)
            inter = iw * ih
            area1 = (current[2] - current[0]) * (current[3] - current[1])
            area2 = (r[2] - r[0]) * (r[3] - r[1])
            union = area1 + area2 - inter
            iou = inter / union if union > 0 else 0
            if iou <= iou_thresh:
                rest.append(r)
        rects = rest
    return keep


def process_image(image_path):
    unet_model, keras_model, class_names, transform, device = load_models()
    original_image = Image.open(image_path).convert('RGB')
    orig_w, orig_h = original_image.size
    input_image = original_image.resize((256, 256), Image.BILINEAR)
    input_tensor = transform(input_image).unsqueeze(0).to(device)
    with torch.no_grad():
        heatmap_pred = unet_model(input_tensor)
    heatmap_np = heatmap_pred.squeeze().cpu().numpy()
    heatmap_resized = cv2.resize(heatmap_np, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    centers, boxes, thresh = find_components(heatmap_resized, min_area=max(20, int((orig_w * orig_h) * 0.0001)))
    detections = non_max_suppression(centers, boxes, iou_thresh=0.3)
    results = []
    for cx, cy, bbox in detections:
        x1b, y1b, x2b, y2b = bbox
        bw = x2b - x1b
        bh = y2b - y1b
        x1, y1, x2, y2 = clamp_bbox(x1b, y1b, bw, bh, orig_w, orig_h, pad=int(max(bw, bh) * 0.2))
        patch = original_image.crop((x1, y1, x2, y2))
        patch = ImageOps.fit(patch, (224, 224), Image.Resampling.LANCZOS)
        patch_array = np.asarray(patch).astype(np.float32)
        if keras_model is not None:
            normalized_array = (patch_array / 127.5) - 1.0
            data = np.expand_dims(normalized_array, axis=0)
            try:
                prediction = keras_model.predict(data, verbose=0)
                index = int(np.argmax(prediction, axis=1)[0])
                confidence = float(prediction[0][index])
                class_name = class_names[index].split(' ', 1)[-1].strip() if index < len(class_names) else f"class_{index}"
            except Exception:
                class_name = 'unknown'
                confidence = 0.0
        else:
            class_name = 'unknown'
            confidence = 0.0
        distance = math.hypot(cx, cy)
        results.append({'class': class_name, 'confidence': confidence, 'coordinates': (cx, cy), 'bbox': (x1, y1, x2, y2), 'distance': distance})
    results.sort(key=lambda x: x['distance'])
    return results


if __name__ == "__main__":
    results = process_image("dt/Фото 8/WIN_20251107_18_28_49_Pro.jpg")
    for item in results:
        print(f"Class: {item['class']}, Confidence: {item['confidence']:.4f}, Coordinates: {item['coordinates']}, BBox: {item['bbox']}")
