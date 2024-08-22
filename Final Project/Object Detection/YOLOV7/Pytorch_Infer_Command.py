import os

# Set the path to the target image
image_path = r"C:\Users\Elie_\Desktop\VS CODE PROJECTS\Final Project Inmind\dataset\test\Images\rgb_0165.png"  

# Change to the YOLOv7 directory
os.chdir(r"C:\Users\Elie_\Desktop\VS CODE PROJECTS\Final Project Inmind\YOLOV7\yolov7")

# Run inference using the trained model
os.system(f'python detect.py --weights runs/train/yolov7-w6-custom/weights/best.pt --img 640 --conf 0.4 --source "{image_path}"')

# Change back to the project directory
os.chdir(r"C:\Users\Elie_\Desktop\VS CODE PROJECTS\Final Project Inmind")
