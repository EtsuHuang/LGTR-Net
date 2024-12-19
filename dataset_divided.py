import os
import shutil
import random


def split_dataset(input_dir, output_dir, test_ratio=0.3, random_seed=42):
    random.seed(random_seed)
    train_dir = os.path.join(output_dir, 'train')
    test_dir = os.path.join(output_dir, 'test')

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    for category in os.listdir(input_dir):
        category_path = os.path.join(input_dir, category)
        if os.path.isdir(category_path):
            files = os.listdir(category_path)
            random.shuffle(files)
            test_size = int(len(files) * test_ratio)
            train_size = len(files) - test_size
            os.makedirs(os.path.join(train_dir, category), exist_ok=True)
            os.makedirs(os.path.join(test_dir, category), exist_ok=True)
            for file in files[:train_size]:
                shutil.copy(os.path.join(category_path, file), os.path.join(train_dir, category, file))
            for file in files[train_size:]:
                shutil.copy(os.path.join(category_path, file), os.path.join(test_dir, category, file))

input_directory = 'E:\\classification-pytorch-main\\classification-pytorch-main\\SSSC_10'  # 输入路径
output_directory = 'E:\\classification-pytorch-main\\classification-pytorch-main\\Divided_Dataset_SSSC'  # 输出路径

split_dataset(input_directory, output_directory, test_ratio=0.2, random_seed=42)
