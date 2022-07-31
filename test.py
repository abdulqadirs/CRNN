import os
from pathlib import Path

images_dir = Path('dataset/upti-1/ligature_undegraded/')
labels_dir = Path('dataset/upti-1/groundtruth')

image_files = os.listdir(images_dir)
label_files = os.listdir(labels_dir)

unique_chars = []
for i, label_file in enumerate(label_files):
    label_path = Path(labels_dir / label_file)
    with open(label_path, 'r') as txt_file:
        text = txt_file.readline().strip()
        for char in text:
            if char not in unique_chars:
                unique_chars.append(char)
print(unique_chars)
# for char in unique_chars:
#     print(char, type(char))
