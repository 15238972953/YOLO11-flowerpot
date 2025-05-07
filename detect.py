# -*- coding: utf-8 -*-


from ultralytics import YOLO
import cv2

if __name__ == '__main__':

    # Load a model
    model = YOLO(model=r'runs\\train\\exp2\\weights\\best.pt')  
    model.predict(source='flower_data\\val\\images\\*.jpg',
                  save=True,
                  show=False,
                  )

