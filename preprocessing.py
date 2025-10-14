import os


file_names = os.listdir("train_imgs")
path = "train_labels"
for file_name in file_names:
    with open(f"{path}/{file_name[:-4]}.txt", "w", encoding="utf-8") as file:
        file.write(str(int(file_name[0]) - 1))
print("preprocessing finished")
