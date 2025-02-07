import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO



if __name__ == '__main__':
    model = YOLO('ultralytics/cfg/models/mamba-yolo/Mamba-YOLO-B.yaml')
    # model.load('yolov8n.pt') # loading pretrain weights
    model.train(data='dataset/data.yaml',
                cache=False,
                imgsz=640,
                epochs=200,
                batch=8,
                close_mosaic=0,
                workers=8,
                # device='0',
                optimizer='SGD', # using SGD
                # patience=0, # close earlystop
                # resume=True, 
                # amp=False, # close amp
                # fraction=0.2,
                project='runs/trainmambaYOLO',
                name='exp',
                )
