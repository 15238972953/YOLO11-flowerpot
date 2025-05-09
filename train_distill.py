import warnings
warnings.filterwarnings('ignore')
import argparse, yaml, copy
from ultralytics.models.yolo.detect.distill import DetectionDistiller

if __name__ == '__main__':
    param_dict = {
        # student
        'model': 'ultralytics/cfg/models/11/yolov11-flower.yaml',
        'data':'ultralytics/cfg/datasets/flower.yaml',
        'imgsz': 640,
        'epochs': 250,
        'batch': 16,
        'workers': 8,
        'cache': False,
        'optimizer': 'SGD',
        'device': '0',
        'close_mosaic': 10,
        'amp': False, # 如果蒸馏损失为nan，请把amp设置为False
        'project':'runs/train',
        'name':'exp_distill',
        
        # teacher
        'teacher_weights': 'runs\\train\\exp_250_m\\weights\\best.pt',
        'teacher_cfg': 'ultralytics\\cfg\\models\\11\\yolo11m.yaml',
        'kd_loss_type': 'all',
        'kd_loss_decay': 'constant',
        
        'logical_loss_type': 'BCKD',
        'logical_loss_ratio': 0.8,
        
        'teacher_kd_layers': '16,19,22',
        'student_kd_layers': '16,19,22',
        'feature_loss_type': 'cwd',
        'feature_loss_ratio': 0.5
    }
    
    model = DetectionDistiller(overrides=param_dict)
    model.distill()