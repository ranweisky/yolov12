import os
import cv2
import json
import random
import shutil
import numpy as np
from pathlib import Path
from tqdm import tqdm

class DatasetProcessor:
    def __init__(self, json_path, src_dir, dst_dir, target_size=640, overlap=100):
        """
        初始化数据集处理器
        
        参数:
            json_path: VIA格式的JSON标注文件路径
            src_dir: 源图片目录
            dst_dir: 目标数据集目录
            target_size: 切片大小
            overlap: 重叠像素数
        """
        self.json_path = os.path.expanduser(json_path)
        self.src_dir = Path(os.path.expanduser(src_dir))
        self.dst_dir = Path(os.path.expanduser(dst_dir))
        self.target_size = target_size
        self.overlap = overlap
        
        # 创建临时目录和最终目录
        self.temp_dir = self.dst_dir / 'temp'
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        (self.temp_dir / 'labels').mkdir(exist_ok=True)
        
        # 创建训练、验证、测试目录
        for split in ['train', 'val', 'test']:
            (self.dst_dir / split / 'images').mkdir(parents=True, exist_ok=True)
            (self.dst_dir / split / 'labels').mkdir(parents=True, exist_ok=True)

    def convert_via_to_yolo(self):
        """将VIA格式的JSON转换为YOLO格式的标签"""
        print('开始转换JSON标注...')
        
        with open(self.json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        metadata = data['_via_img_metadata']
        processed_files = []
        
        for filename, img_data in tqdm(metadata.items()):
            try:
                image_name = img_data['filename']
                regions = img_data['regions']
                
                # 读取图片获取尺寸
                img_path = self.src_dir / image_name
                if not img_path.exists():
                    print(f'警告：找不到图片 {image_name}')
                    continue
                    
                img = cv2.imread(str(img_path))
                if img is None:
                    print(f'警告：无法读取图片 {image_name}')
                    continue
                
                height, width = img.shape[:2]
                
                # 创建标签文件
                label_path = self.temp_dir / 'labels' / f'{Path(image_name).stem}.txt'
                
                with open(label_path, 'w') as f:
                    for region in regions:
                        shape_attrs = region['shape_attributes']
                        region_attrs = region['region_attributes']
                        
                        if shape_attrs['name'] == 'rect':
                            # 获取边界框坐标
                            x = shape_attrs['x']
                            y = shape_attrs['y']
                            w = shape_attrs['width']
                            h = shape_attrs['height']
                            
                            # 确定类别
                            cls_id = None
                            if 'flower' in region_attrs and region_attrs['flower'] == '1':
                                cls_id = 0
                            elif 'tomato' in region_attrs and region_attrs['tomato'] == '1':
                                cls_id = 1
                            
                            if cls_id is not None:
                                # 转换为YOLO格式
                                x_center = (x + w/2) / width
                                y_center = (y + h/2) / height
                                w = w / width
                                h = h / height
                                
                                f.write(f'{cls_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n')
                
                processed_files.append(image_name)
                
            except Exception as e:
                print(f'处理 {filename} 时出错: {str(e)}')
        
        print(f'标注转换完成，共处理 {len(processed_files)} 个文件')
        return processed_files

    def split_images_and_labels(self, processed_files):
        """将图片切分成小块并调整对应的标签"""
        print('\n开始切分图片和标签...')
        split_files = []
        
        for image_name in tqdm(processed_files):
            img_path = self.src_dir / image_name
            label_path = self.temp_dir / 'labels' / f'{Path(image_name).stem}.txt'
            
            # 读取图片和标签
            img = cv2.imread(str(img_path))
            height, width = img.shape[:2]
            
            with open(label_path, 'r') as f:
                labels = []
                for line in f:
                    cls_id, x_center, y_center, w, h = map(float, line.strip().split())
                    # 转换回像素坐标
                    x_center = int(x_center * width)
                    y_center = int(y_center * height)
                    w = int(w * width)
                    h = int(h * height)
                    labels.append([cls_id, x_center, y_center, w, h])
            
            # 计算切分网格
            stride = self.target_size - self.overlap
            x_splits = range(0, width - self.overlap, stride)
            y_splits = range(0, height - self.overlap, stride)
            
            # 添加最后的切片位置
            if width % stride != 0:
                x_splits = list(x_splits) + [width - self.target_size]
            if height % stride != 0:
                y_splits = list(y_splits) + [height - self.target_size]
            
            # 切分图片和标签
            for i, x_start in enumerate(x_splits):
                for j, y_start in enumerate(y_splits):
                    x_end = min(x_start + self.target_size, width)
                    y_end = min(y_start + self.target_size, height)
                    x_start = min(x_start, width - self.target_size)
                    y_start = min(y_start, height - self.target_size)
                    
                    # 切分图片
                    split_img = img[y_start:y_end, x_start:x_end]
                    
                    # 处理标签
                    split_labels = []
                    for cls_id, x_center, y_center, w, h in labels:
                        # 检查边界框是否在切片内
                        box_x1 = x_center - w/2
                        box_y1 = y_center - h/2
                        box_x2 = x_center + w/2
                        box_y2 = y_center + h/2
                        
                        if (box_x2 > x_start and box_x1 < x_end and 
                            box_y2 > y_start and box_y1 < y_end):
                            # 调整坐标到切片坐标系
                            new_x_center = x_center - x_start
                            new_y_center = y_center - y_start
                            
                            # 确保坐标在范围内
                            new_x_center = min(max(new_x_center, 0), self.target_size)
                            new_y_center = min(max(new_y_center, 0), self.target_size)
                            
                            # 转换回归一化坐标
                            new_x_center = new_x_center / self.target_size
                            new_y_center = new_y_center / self.target_size
                            new_w = min(w / self.target_size, 1.0)
                            new_h = min(h / self.target_size, 1.0)
                            
                            split_labels.append([cls_id, new_x_center, new_y_center, new_w, new_h])
                    
                    # 只保存包含目标的切片
                    if split_labels:
                        new_name = f'{Path(image_name).stem}_{i}_{j}.jpg'
                        split_files.append({
                            'image': new_name,
                            'labels': split_labels
                        })
                        
                        # 保存切分的图片和标签
                        cv2.imwrite(str(self.temp_dir / new_name), split_img)
                        
                        with open(self.temp_dir / 'labels' / f'{Path(new_name).stem}.txt', 'w') as f:
                            for label in split_labels:
                                f.write(f'{int(label[0])} {label[1]:.6f} {label[2]:.6f} {label[3]:.6f} {label[4]:.6f}\n')
        
        print(f'切分完成，生成了 {len(split_files)} 个有效切片')
        return split_files

    def create_dataset_splits(self, split_files, train_ratio=0.8, val_ratio=0.1):
        """将数据集分割为训练集、验证集和测试集"""
        print('\n开始分割数据集...')
        
        # 随机打乱文件列表
        random.shuffle(split_files)
        
        # 计算分割点
        total = len(split_files)
        train_size = int(total * train_ratio)
        val_size = int(total * val_ratio)
        
        # 分割数据集
        train_files = split_files[:train_size]
        val_files = split_files[train_size:train_size + val_size]
        test_files = split_files[train_size + val_size:]
        
        # 复制文件到对应目录
        splits = {
            'train': train_files,
            'val': val_files,
            'test': test_files
        }
        
        for split_name, files in splits.items():
            print(f'处理{split_name}集，{len(files)}个文件')
            for file_info in files:
                image_name = file_info['image']
                
                # 复制图片
                shutil.copy2(
                    self.temp_dir / image_name,
                    self.dst_dir / split_name / 'images' / image_name
                )
                
                # 复制标签
                shutil.copy2(
                    self.temp_dir / 'labels' / f'{Path(image_name).stem}.txt',
                    self.dst_dir / split_name / 'labels' / f'{Path(image_name).stem}.txt'
                )
        
        # 创建data.yaml
        yaml_content = f"""path: {self.dst_dir}
train: train/images
val: val/images
test: test/images
nc: 2
names: ['flower', 'tomato']"""
        
        with open(self.dst_dir / 'data.yaml', 'w') as f:
            f.write(yaml_content)
        
        # 清理临时目录
        shutil.rmtree(self.temp_dir)
        
        print('\n数据集处理完成！')
        print(f'训练集: {len(train_files)} 个文件')
        print(f'验证集: {len(val_files)} 个文件')
        print(f'测试集: {len(test_files)} 个文件')

def main():
    # 设置随机种子
    random.seed(42)
    
    # 配置参数
    json_path = '~/autodl-tmp/datasets/tomatoSrc/tomato王云1210.json'
    src_dir = '~/autodl-tmp/datasets/tomatoSrc'
    dst_dir = '~/autodl-tmp/datasets/tomato'
    
    # 创建处理器
    processor = DatasetProcessor(json_path, src_dir, dst_dir)
    
    # 执行处理流程
    processed_files = processor.convert_via_to_yolo()
    split_files = processor.split_images_and_labels(processed_files)
    processor.create_dataset_splits(split_files)

if __name__ == '__main__':
    main()
