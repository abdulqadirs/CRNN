import os
from pathlib import Path

images_dir = Path('dataset/upti-1/ligature_undegraded/')
labels_dir = Path('dataset/upti-1/groundtruth')

image_files = os.listdir(images_dir)
label_files = os.listdir(labels_dir)

# unique_chars = []
# for i, label_file in enumerate(label_files):
#     label_path = Path(labels_dir / label_file)
#     with open(label_path, 'r') as txt_file:
#         text = txt_file.readline().strip()
#         for char in text:
#             if char not in unique_chars:
#                 unique_chars.append(char)
# print(unique_chars)
# for char in unique_chars:
#     print(char, type(char))
chars  = ['ظ', 'ف', 'ر', ' ', 'ا', 'ق', 'ب', 'ل', '(', 'ڈ', 'ی', 'ن', 'ک', 'ٹ', 'آ', 'س', ')',
            'ے', 'م', 'ت', 'ش', 'د', 'و', 'پ', 'خ', 'ں', 'ھ', 'ہ', 'ض', 'ع', 'ئ', 'ج', 'چ', 'ح', '۔', 'گ',
            'ٴ', 'ذ', '4', '2', '8', '7', 'ز', '0', '1', '،', 'ث', 'ط', '/', 'ص', '5', 'غ', 'ؤ', '3', 'ڑ', 
            '6', '؟', ',', '9', '.', '\ue000', 'َ', '‘', 'ٰ', '…', ':', '”', '“', 'ء', 'J', 'E', '-', 'ً', 'ُ',
            'ﷺ', 'M', 'r', 'G', 'o', 'b', 'R', 'u', 'f', 'e', 'l', 'ه', 'C', 'a', 'ژ', 'ِ', 'w', 'p', 'm',
            'n', 'c', 'k', 'T', 'P', '"', 'i', 'd', 'K', 'S', 'I', 'g', '#', 'ّ', 'V', '’', 'h', 't', 'B',
            'N', 'ْ', 'أ', '%', 'D', '+', 'O', 'A', '!', 'L', 'ٓ', 'W', '\ue002', 'ة', 'F', 'v', 'z', 's', 
            'Y', 'U', 'x', "'", '؛']
