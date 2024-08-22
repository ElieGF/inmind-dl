import os

# Change to the YOLOv7 directory
os.chdir("C:\\Users\\Elie_\\Desktop\\VS CODE PROJECTS\\Final Project Inmind\\YOLOV7\\yolov7")


# Run training command
os.system('python train_aux.py --workers 8 --device 0 --batch-size 8 --data data/Custom_Dataset.yaml --img 640 640 --cfg cfg/training/yolov7-w6-custom.yaml --weights "yolov7-w6_training.pt" --name yolov7-w6-custom --hyp data/hyp.scratch.custom.yaml --epochs 100')

# Change back to the project directory
os.chdir("C:\\Users\\Elie_\\Desktop\\VS CODE PROJECTS\\Final Project Inmind")