import os
from keras.models import load_model
from PIL import Image, ImageOps, ImageFilter
import numpy as np

np.set_printoptions(suppress=True)


def _crop_object(image):
    # Размытие для уменьшения шумов
    blurred = image.filter(ImageFilter.GaussianBlur(2))
    blurred_array = np.array(blurred)

    # Пороговое значение для выделения не‑белых пикселей
    mask = np.all(blurred_array > 100, axis=-1)  # True для "белых"
    coords = np.nonzero(~mask)  # Ищем не-белые пиксели

    if len(coords[0]) == 0:
        return image

    top, bottom = np.min(coords[0]), np.max(coords[0])
    left, right = np.min(coords[1]), np.max(coords[1])

    left = max(0, left - 5)
    top = max(0, top - 5)
    right = min(image.width, right + 5)
    bottom = min(image.height, bottom + 5)
    cropped = image.crop((left, top, right, bottom))
    return cropped

def model_gala_net(path):

    model = load_model("converted_keras/keras_model.h5", compile=False)

    class_names = open("converted_keras/labels.txt", "r").readlines()

    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    image = Image.open(path).convert("RGB")

    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    image_array = np.asarray(image)

    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    data[0] = normalized_image_array

    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]
    return class_name, confidence_score



def main(path_to_img):
    img = Image.open(path_to_img).convert("RGB")
    img = _crop_object(img)
    path = "converted_keras/img.jpg"
    img.save(path)
    class_name, confidence_score = model_gala_net(path)
    return int(class_name.split("\n")[-2][-1])


if __name__ == '__main__':
    validation = True
    if validation:
        folders = os.listdir('датасет 3 (1)')
        max_len = 0
        cnt = 0
        for f in folders:
            if "." not in f:
                files = os.listdir(f'датасет 3 (1)/{f}')
                img = ""
                label = 0
                for file in files:
                    if file.endswith('.jpg'):
                        img = f"датасет 3 (1)/{f}/{file}"
                    if file.endswith('.txt'):
                        with open(f"датасет 3 (1)/{f}/{file}", "r", encoding="utf-8") as text_file:
                            label = int(text_file.read().strip("\n"))

                class_name = main(img)

                if class_name == label:
                    cnt += 1
                max_len += 1
        print(cnt / max_len)
    else:
        print(main("датасет/Диод кр син зел желт/WIN_20251028_17_56_33_Pro_obj.png"))