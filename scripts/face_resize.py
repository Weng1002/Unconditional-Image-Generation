import os
import cv2
from tqdm import tqdm

input_dir = "../2023_正妹_filterface"
output_dir = "../datasets/2023_正妹_resized"
os.makedirs(output_dir, exist_ok=True)

# Resize 尺寸
target_size = (64, 64)

for fname in tqdm(os.listdir(input_dir)):
    if not fname.lower().endswith(".png"):
        continue

    img_path = os.path.join(input_dir, fname)
    img = cv2.imread(img_path)
    if img is None:
        continue

    resized_img = cv2.resize(img, target_size)
    save_path = os.path.join(output_dir, fname)
    cv2.imwrite(save_path, resized_img)
