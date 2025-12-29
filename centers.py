import cv2
import numpy as np
import os


def process_image(image_path, output_folder):
    img = cv2.imread(image_path)
    if img is None:
        return

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

        cropped = cv2.resize(cropped, (70, 70))
        resized = cv2.resize(cropped, (40, 40), interpolation=cv2.INTER_AREA)

        cv2.imwrite(f'{output_folder}/part_{i:03d}.png', resized)

if __name__ == "__main__":
    process_image("датасет/0/WIN_20251105_15_50_05_Pro.jpg", "output_parts")