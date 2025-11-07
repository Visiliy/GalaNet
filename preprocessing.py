import os


dir_names = os.listdir("датасет")
for i, name in enumerate(dir_names):
    files = os.listdir("датасет/" + name)
    for file in files:
        with open(f"датасет/{name}/{file[:-4]}_text.txt", "w", encoding="utf-8") as f:
            f.write(str(i))

print("Ok")