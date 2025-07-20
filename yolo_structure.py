import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

# === CONFIG ===
csv_path = "kag_straw/strawberries.csv"
image_folder = "kag_straw/boxes"
label_folder = "labels"
output_image_dir = "dataset/images"
output_label_dir = "dataset/labels"
split_ratio = 0.2  # 20% val

# === STEP 1: READ EXCEL ===
df = pd.read_csv(csv_path)
image_list = df['image_name'].tolist()

# === STEP 2: SPLIT ===
train_files, val_files = train_test_split(image_list, test_size=split_ratio, random_state=42)

# === STEP 3: FUNCTION TO MOVE FILES ===
def move_files(file_list, split_type):
    os.makedirs(os.path.join(output_image_dir, split_type), exist_ok=True)
    os.makedirs(os.path.join(output_label_dir, split_type), exist_ok=True)

    for file in file_list:
        image_src = os.path.join(image_folder, os.path.basename(file))
        label_src = os.path.join(label_folder, os.path.splitext(os.path.basename(file))[0] + ".txt")

        image_dst = os.path.join(output_image_dir, split_type, os.path.basename(file))
        label_dst = os.path.join(output_label_dir, split_type, os.path.basename(label_src))

        if os.path.exists(image_src):
            shutil.copy(image_src, image_dst)
        else:
            print(f"Image not found: {image_src}")

        if os.path.exists(label_src):
            shutil.copy(label_src, label_dst)
        else:
            print(f"Label not found: {label_src}")

# === STEP 4: MOVE ===
move_files(train_files, "train")
move_files(val_files, "val")

print("✅ Split completed and files moved.")
