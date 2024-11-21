import os
import nibabel as nib
import imageio
import numpy as np
from tqdm import tqdm

from skimage.transform import resize

from glob import glob
from PIL import Image


# 3D数据切片为2D数据
def nii_to_image(filepath, imgfile, output_size=(256, 256)):
    filenames = os.listdir(filepath)  # 读取nii文件夹
    with tqdm(total=len(filenames)) as pbar:
        for f in filenames:
            if f[-7:] != ".nii.gz":
                continue
            img_path = os.path.join(filepath, f)
            img = nib.load(img_path)  # 读取nii
            img_fdata = img.get_fdata(dtype=np.float32)  # 读取为float32
            fname = f.replace('.nii.gz', '')  # 去掉nii的后缀名
            img_f_path = os.path.join(imgfile, fname)
            # 创建nii对应的图像的文件夹
            if not os.path.exists(img_f_path):
                os.mkdir(img_f_path)  # 新建文件夹

            # 将归一化的浮点数数据转换为8位整数数据
            img_fdata = (img_fdata * 255).astype(np.uint8)

            # 开始转换为图像
            (x, y, z) = img.shape
            for i in range(z):  # z是图像的序列
                slice = img_fdata[:, :, i]  # 选择哪个方向的切片都可以

                # 调整切片大小并保持长宽比
                resized_slice = resize(slice, output_size, mode='constant', anti_aliasing=True)
                resized_slice = (resized_slice * 255).astype(np.uint8)  # 调整后的图像重新缩放到0-255

                # 保存图像，使用PNG格式
                imageio.imwrite(os.path.join(img_f_path, '{}.png'.format(i)), resized_slice)
            pbar.update(1)


# 删除标签为全黑的图像以及对应的原图
def delete_black_labels_and_originals(labelsTr_folder, imagesTr_folder):
    """
    删除指定文件夹中所有全黑的标签文件，并删除与之对应的原图文件。

    :param labelsTr_folder: 标签文件所在的文件夹路径
    :param imagesTr_folder: 原图文件所在的文件夹路径
    """
    # 遍历labelsTr文件夹中的所有子文件夹
    for subject_folder in os.listdir(labelsTr_folder):
        subject_labels_path = os.path.join(labelsTr_folder, subject_folder)

        # 检查是否为文件夹
        if os.path.isdir(subject_labels_path):
            # 获取所有标签文件的路径
            label_files = glob(os.path.join(subject_labels_path, '*.png'))

            # 对应的原图文件夹路径,原图文件夹有_0000的后缀
            subject_images_path = os.path.join(imagesTr_folder, subject_folder + "_0000")

            # 遍历每个标签文件
            for label_file in label_files:
                # 打开图片并检查是否全黑
                with Image.open(label_file) as img:
                    # 将图片转换为灰度模式以便检查
                    img_gray = img.convert('L')

                    # 计算图片的像素值总和
                    total_pixels = img_gray.size[0] * img_gray.size[1]
                    sum_pixels = sum(img_gray.getdata())

                    # 如果图片全黑，则像素值总和为0
                    if sum_pixels == 0:
                        print(f"Deleting {label_file}...")

                        # 删除标签文件
                        os.remove(label_file)

                        # 构建对应原图的文件名
                        original_file = os.path.splitext(os.path.basename(label_file))[0] + '.png'
                        original_path = os.path.join(subject_images_path, original_file)

                        # 删除原图
                        if os.path.exists(original_path):
                            os.remove(original_path)
                            print(f"Also deleting original image {original_path}...")
                        else:
                            print(f"Original image {original_path} not found.")


def delete_images_without_labels(labelsTr_folder, imagesTr_folder):
    """
    删除没有对应标签文件的原图文件。

    :param labelsTr_folder: 标签文件所在的文件夹路径
    :param imagesTr_folder: 原图文件所在的文件夹路径
    """
    # 遍历imagesTr文件夹中的所有子文件夹
    for images_subfolder in os.listdir(imagesTr_folder):
        images_subfolder_path = os.path.join(imagesTr_folder, images_subfolder)

        # 检查是否为文件夹
        if os.path.isdir(images_subfolder_path):
            # 提取原图子文件夹的编号
            subject_number = images_subfolder.split('_')[1].split('_')[0]

            # 构建对应的标签文件夹路径
            labels_subfolder = f'FLARE22_{subject_number}'
            labels_subfolder_path = os.path.join(labelsTr_folder, labels_subfolder)

            # 检查标签文件夹是否存在
            if not os.path.exists(labels_subfolder_path):
                print(f"No corresponding label folder found for {images_subfolder_path}. Skipping...")
                continue

            # 获取所有原图文件的路径
            image_files = glob(os.path.join(images_subfolder_path, '*.png'))

            # 获取所有标签文件的路径
            label_files = glob(os.path.join(labels_subfolder_path, '*.png'))

            # 创建一个集合来存储标签文件的基名
            label_basenames = set(os.path.splitext(os.path.basename(label_file))[0] for label_file in label_files)

            # 遍历每个原图文件
            for image_file in image_files:
                # 构建对应标签文件的基名
                image_basename = os.path.splitext(os.path.basename(image_file))[0]

                # 如果原图文件没有对应的标签文件
                if image_basename not in label_basenames:
                    print(f"Deleting {image_file} as it has no corresponding label file...")
                    os.remove(image_file)

# 统计原图和标签每个子文件夹中文件的数量
def count_files_and_print(labelsTr_folder, imagesTr_folder):
    """
    统计原图和标签文件夹中每个子文件夹中的文件数量，并将它们一一对应输出。

    :param labelsTr_folder: 标签文件所在的文件夹路径
    :param imagesTr_folder: 原图文件所在的文件夹路径
    """
    # 获取所有标签文件夹的名称
    labels_subfolders = [f for f in os.listdir(labelsTr_folder) if os.path.isdir(os.path.join(labelsTr_folder, f))]

    # 获取所有原图文件夹的名称
    images_subfolders = [f for f in os.listdir(imagesTr_folder) if os.path.isdir(os.path.join(imagesTr_folder, f))]

    # 创建一个字典来存储标签文件夹和对应的原图文件夹
    label_to_image = {}

    # 遍历标签文件夹
    for labels_subfolder in labels_subfolders:
        # 提取标签文件夹的编号
        label_number = labels_subfolder.split('_')[1]

        # 构建对应的原图文件夹名称
        image_subfolder = f'FLARE22_{label_number}_0000'

        # 检查原图文件夹是否存在
        if image_subfolder in images_subfolders:
            label_to_image[labels_subfolder] = image_subfolder

    num = 0
    total_label_num = 0
    num_image = []
    # 统计每个子文件夹中的文件数量并打印
    for label_subfolder, image_subfolder in label_to_image.items():
        label_subfolder_path = os.path.join(labelsTr_folder, label_subfolder)
        image_subfolder_path = os.path.join(imagesTr_folder, image_subfolder)

        # 获取标签文件的数量
        label_files = glob(os.path.join(label_subfolder_path, '*.png'))
        num_label_files = len(label_files)

        # 获取原图文件的数量
        image_files = glob(os.path.join(image_subfolder_path, '*.png'))
        num_image_files = len(image_files)

        print(f"Label Folder: {label_subfolder_path}, Number of Files: {num_label_files}")
        print(f"Image Folder: {image_subfolder_path}, Number of Files: {num_image_files}")
        total_label_num += num_label_files

        if num_label_files != num_image_files:
            num += 1
            num_image.append(image_subfolder_path)
            print("数量不等:", image_subfolder_path)

    print("总统计：-----------------------------------------")
    print("标签总数量：", total_label_num)
    print("不相同文件夹数量：", num)
    for i in num_image:
        print("不相同文件夹：", i)



if __name__ == '__main__':
    # 3D数据切片为2D数据
    # file_name = r"/mnt/d/01_yjh/data/3D/Dataset701_AbdomenCT/imagesVal"
    # img_path = r"/mnt/d/01_yjh/data/2D/Dataset701_AbdomenCT/imagesVal"
    # nii_to_image(file_name, img_path, output_size=(256, 256))


    # # 删除标签为全黑的图像以及对应的原图
    # imagesTr_folder = r"/mnt/d/01_yjh/data/2D/Dataset701_AbdomenCT/imagesVal"
    # labelsTr_folder = r"/mnt/d/01_yjh/data/2D/Dataset701_AbdomenCT/labelsVal"
    # delete_black_labels_and_originals(labelsTr_folder, imagesTr_folder)


    # 统计原图和标签每个子文件夹中文件的数量
    imagesTr_folder = '/mnt/d/01_yjh/data/2D/Dataset701_AbdomenCT/imagesVal'
    labelsTr_folder = '/mnt/d/01_yjh/data/2D/Dataset701_AbdomenCT/labelsVal'
    count_files_and_print(labelsTr_folder, imagesTr_folder)


    # # 删除没有标签的原图
    # imagesTr_folder = '/mnt/d/01_yjh/data/2D/Dataset701_AbdomenCT/imagesTr'
    # labelsTr_folder = '/mnt/d/01_yjh/data/2D/Dataset701_AbdomenCT/labelsTr'
    # delete_images_without_labels(labelsTr_folder, imagesTr_folder)