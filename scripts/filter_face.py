import os
import cv2
import numpy as np
from tqdm import tqdm
from insightface.app import FaceAnalysis

# 模糊判斷
def is_blurry(img, threshold=50):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return lap_var < threshold

# 雙眼亮度判斷（濾掉遮眼、瀏海、墨鏡）
def eye_region_brightness(img, left_eye, right_eye):
    patch_size = 10  # pixel
    eyes = [left_eye, right_eye]
    brightness = []
    for eye in eyes:
        x, y = int(eye[0]), int(eye[1])
        patch = img[max(0, y - patch_size):y + patch_size, max(0, x - patch_size):x + patch_size]
        if patch.size == 0:
            return 0  # 若 patch 無效則視為太暗
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        brightness.append(np.mean(gray))
    return np.mean(brightness)

input_dir = "../raw_images/2025_正妹_images"
output_dir = "../raw_images/2025_正妹_filterface"
os.makedirs(output_dir, exist_ok=True)

# 初始化 InsightFace
app = FaceAnalysis(name='buffalo_l')  
app.prepare(ctx_id=0)

# 條件設定
MIN_RES = (512, 512)
MIN_FACE_SIZE = 100

for fname in tqdm(os.listdir(input_dir)):
    img_path = os.path.join(input_dir, fname)
    img = cv2.imread(img_path)
    if img is None:
        continue

    h, w = img.shape[:2]
    if h < MIN_RES[1] or w < MIN_RES[0]:
        continue

    faces = app.get(img)
    for i, face in enumerate(faces):
        pitch, yaw, roll = face.pose
        if abs(yaw) > 20 or abs(pitch) > 20:
            continue  # 過側或仰頭低頭
        
        x1, y1, x2, y2 = face.bbox.astype(int)

        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        face_size = max(x2 - x1, y2 - y1)
        square_size = int(face_size * 1.6)  # 可以調整 1.6 → 大小留白空間

        x1_new = max(cx - square_size // 2, 0)
        y1_new = max(cy - square_size // 2, 0)
        x2_new = min(cx + square_size // 2, w)
        y2_new = min(cy + square_size // 2, h)
        
        if x2_new <= x1_new or y2_new <= y1_new:
            continue

        face_crop = img[y1_new:y2_new, x1_new:x2_new]
        if face_crop.size == 0:
            continue

        # 雙眼距離過短 → 側臉（或單眼）
        left_eye, right_eye = face.kps[0], face.kps[1]
        eye_dist = np.linalg.norm(left_eye - right_eye)
        if eye_dist < 15:
            continue

        # 雙眼亮度過暗 → 遮眼（瀏海、墨鏡、閉眼）
        brightness = eye_region_brightness(img, left_eye, right_eye)
        if brightness < 60:
            continue

        # 模糊判斷（Laplacian）
        if is_blurry(face_crop, threshold=50):
            continue

        save_path = os.path.join(output_dir, f"{os.path.splitext(fname)[0]}_face{i}.png")
        cv2.imwrite(save_path, face_crop)