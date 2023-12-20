import os
import shutil

def extract_and_rename_images(src_folder, dest_folder):
    # 创建目标文件夹
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    # 遍历源文件夹中的每个子文件夹
    for root, dirs, files in os.walk(src_folder):
        for file in files:
            # 获取文件的完整路径
            src_path = os.path.join(root, file)

            # 构建目标文件路径
            dest_path = os.path.join(dest_folder, file)

            # 如果目标文件已经存在，则生成新的文件名
            count = 1
            while os.path.exists(dest_path):
                filename, extension = os.path.splitext(file)
                new_filename = f"{filename}_{count}{extension}"
                dest_path = os.path.join(dest_folder, new_filename)
                count += 1

            # 复制文件到目标文件夹
            shutil.copy(src_path, dest_path)
            print(f"复制文件: {src_path} 到 {dest_path}")

if __name__ == "__main__":
    # 指定源文件夹和目标文件夹
    men_folder = "img/MEN"
    women_folder = "img/WOMEN"
    dest_folder = "images"

    # 提取并重命名MEN文件夹中的图片
    extract_and_rename_images(men_folder, dest_folder)

    # 提取并重命名WOMEN文件夹中的图片
    extract_and_rename_images(women_folder, dest_folder)
