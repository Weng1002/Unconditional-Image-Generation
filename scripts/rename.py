import os

# 設定圖片所在的資料夾
folder_path = '../generated_images'

# 取得所有檔案並排序（可選）
file_list = sorted(os.listdir(folder_path))

# 過濾出圖片檔案（可依需求修改）
image_extensions = ['.png']

# 逐一重新命名
count = 1
for filename in file_list:
    ext = os.path.splitext(filename)[1].lower()
    if ext in image_extensions:
        new_name = f'{count:05d}{ext}'
        src = os.path.join(folder_path, filename)
        dst = os.path.join(folder_path, new_name)
        os.rename(src, dst)
        count += 1

print(f"已重新命名 {count-1} 張圖片")
