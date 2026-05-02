import os
import shutil
from sklearn.model_selection import train_test_split

SOURCE_DIR = '/home/ubuntu/Thompson/DS_Final_Project/crops_image'
ROOT_DIR = '/home/ubuntu/Thompson/DS_Final_Project/dataset'
IMG_DIR = os.path.join(ROOT_DIR, 'images')

# 建立 train/val/test 資料夾結構
for subset in ['train', 'val', 'test']:
    os.makedirs(os.path.join(IMG_DIR, subset), exist_ok=True)

# 處理每個類別資料夾
for class_folder in os.listdir(SOURCE_DIR):
    class_path = os.path.join(SOURCE_DIR, class_folder)
    if not os.path.isdir(class_path):
        continue

    all_images = sorted([f for f in os.listdir(class_path) if f.endswith('.png')])
    full_paths = [os.path.join(class_path, f) for f in all_images]

    # 切分圖片(8:1:1)
    train_imgs, temp_imgs = train_test_split(full_paths, test_size=0.2, random_state=42)
    val_imgs, test_imgs = train_test_split(temp_imgs, test_size=0.5, random_state=42)

    # 建立每個subset下的子類別資料夾
    for subset, subset_imgs in zip(['train', 'val', 'test'], [train_imgs, val_imgs, test_imgs]):
        subset_class_dir = os.path.join(IMG_DIR, subset, class_folder)
        os.makedirs(subset_class_dir, exist_ok=True)

        for src_path in subset_imgs:
            fname = os.path.basename(src_path)
            dst_path = os.path.join(subset_class_dir, fname)
            shutil.copy(src_path, dst_path)

print("資料切分完成，結構如下：")
print("dataset/images/train/<class>/*.png")
print("dataset/images/val/<class>/*.png")
print("dataset/images/test/<class>/*.png")