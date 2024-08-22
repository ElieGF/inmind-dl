import os

# Path to custom dataset YAML file
Custom_Dataset_Path = 'data/Custom_Dataset.yaml'

# Path to the weights file
weights_path = 'runs/train/yolov7-w6-custom/weights/best.pt'

# Change to the YOLOv7 directory
os.chdir("C:\\Users\\Elie_\\Desktop\\VS CODE PROJECTS\\Final Project Inmind\\YOLOV7\\yolov7")

# Run the evaluation command
os.system(f'python test.py --data {Custom_Dataset_Path} --img 640 --batch-size 8 --conf 0.001 --iou 0.5 --device 0 --weights {weights_path} --task test')

# Change back to the project directory
os.chdir("C:\\Users\\Elie_\\Desktop\\VS CODE PROJECTS\\Final Project Inmind")
