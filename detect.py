# -*- coding: utf-8 -*-


from ultralytics import YOLO
import cv2

if __name__ == '__main__':

    # Load a model
    model = YOLO(model=r'runs\\train\\exp-MobileNetV2\\weights\\best.pt')  
    model.predict(source=0,
                  save=False,
                  show=True,
                  )

