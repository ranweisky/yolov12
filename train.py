from ultralytics import YOLO
import argparse
from pathlib import Path
import shutil
import pandas as pd
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True, help='数据配置文件路径')
    parser.add_argument('--weights', type=str, default='yolov12s.pt', help='预训练权重路径')
    parser.add_argument('--epochs', type=int, default=80, help='训练轮数')
    parser.add_argument('--name', type=str, default='train', help='模型输出路径')
    args = parser.parse_args()

    # 加载模型
    model = YOLO(args.weights)

    # 训练模型
    results = model.train(
        data=args.data,
        imgsz=640,
        batch=16,
        epochs=args.epochs,
        patience=50,
        
        # 小目标检测优化
        overlap_mask=True,
        box=4.0,
        cls=0.8,
        dfl=1.0,
        
        # 学习率设置
        lr0=0.001,
        lrf=0.1,
        warmup_epochs=5,
        warmup_momentum=0.8,
        
        # 数据增强
        mosaic=0.8,
        mixup=0.1,
        copy_paste=0.1,
        degrees=5.0,
        translate=0.2,
        scale=0.4,
        fliplr=0.5,
        flipud=0.1,
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.3,
        
        # 其他参数
        optimizer='AdamW',
        weight_decay=0.0005,
        cache=True,
        amp=True,
        
        # 保存设置
        save_period=10,
        save=True,
        
        # 指定模型输出路径
        project='runs/detect',
        name=args.name
    )

    # 选择最佳模型
    model_path = 'runs/detect/' + args.name
    best_epoch, best_metrics = select_best_model(model_path)
    print(f"\n最佳模型在第 {best_epoch} 轮:")
    print(f"mAP50: {best_metrics['metrics/mAP50']:.4f}")
    print(f"mAP50-95: {best_metrics['metrics/mAP50-95']:.4f}")
    print(f"Precision: {best_metrics['metrics/precision']:.4f}")
    print(f"Recall: {best_metrics['metrics/recall']:.4f}")

    # 分析目标大小分布
    size_stats = analyze_detection_sizes(
        f'{model_path}/weights/epoch{best_epoch}.pt',
        args.data
    )
    print("\n目标大小分布:")
    print(f"小目标: {size_stats['small_objects']}")
    print(f"中等目标: {size_stats['medium_objects']}")
    print(f"大目标: {size_stats['large_objects']}")
    
    # Evaluate model performance on the validation set
    metrics = model.val(imgsz=640)

if __name__ == '__main__':
    main()



# Perform object detection on an image

def select_best_model(results_dir):
    """选择最佳模型"""
    results_file = Path(results_dir) / 'results.csv'
    df = pd.read_csv(results_file)
    
    # 计算综合得分，针对小目标检测调整权重
    df['score'] = (
        df['metrics/mAP50-95'] * 0.3 +    # 整体性能
        df['metrics/mAP50'] * 0.4 +       # 基础检测能力，更重要
        df['metrics/precision'] * 0.1 +    # 精确度
        df['metrics/recall'] * 0.2         # 召回率，小目标要重视召回
    )
    
    # 获取最佳epoch
    best_epoch = df['score'].idxmax()
    best_metrics = df.loc[best_epoch]
    
    return best_epoch, best_metrics

def validate_model(model_path, val_data):
    """验证模型性能"""
    model = YOLO(model_path)
    
    # 使用更严格的验证标准
    results = model.val(
        data=val_data,
        imgsz=1280,
        batch=16,
        conf=0.25,       # 降低置信度阈值，提高小目标检测率
        iou=0.6,         # 适当提高IoU阈值
        max_det=300,     # 增加最大检测数量
        plots=True       # 生成评估图表
    )
    
    return results

def analyze_detection_sizes(model_path, val_data):
    """分析检测目标大小分布"""
    model = YOLO(model_path)
    results = model.val(data=val_data)
    
    # 分析检测框大小分布
    boxes = results.boxes
    areas = boxes.xywh[:, 2] * boxes.xywh[:, 3]  # 计算面积
    
    # 统计不同大小目标的检测效果
    small = (areas < 32*32).sum()
    medium = ((areas >= 32*32) & (areas < 96*96)).sum()
    large = (areas >= 96*96).sum()
    
    return {
        'small_objects': small,
        'medium_objects': medium,
        'large_objects': large
    }