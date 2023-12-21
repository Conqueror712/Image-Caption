import os

def rename_images(folder_path):
    # 检查文件夹是否存在
    if not os.path.exists(folder_path):
        print(f"文件夹 '{folder_path}' 不存在。")
        return

    # 获取文件夹下所有文件
    files = os.listdir(folder_path)

    # 迭代处理每个文件
    for index, file_name in enumerate(files):
        # 获取文件的完整路径
        old_path = os.path.join(folder_path, file_name)

        # 构建新的文件名
        new_name = f"train_{index + 1}.jpg"
        # new_name = f"test_{index + 1}.jpg"

        # 构建新的文件路径
        new_path = os.path.join(folder_path, new_name)

        # 重命名文件
        os.rename(old_path, new_path)

        print(f"重命名文件: {file_name} -> {new_name}")

if __name__ == "__main__":
    # 指定图片文件夹的路径
    images_folder_path = "train_images"
    # images_folder_path = "test_images"

    # 调用函数进行重命名
    rename_images(images_folder_path)
