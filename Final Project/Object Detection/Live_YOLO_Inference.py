import cv2
import os

# Set the path to the YOLOv7 directory
yolo_dir = "C:\\Users\\Elie_\\Desktop\\VS CODE PROJECTS\\Final Project Inmind\\YOLOV7\\yolov7"

# Open a video capture 
cap = cv2.VideoCapture(0)  # Replace 0 with video file path if needed

# Ensure the capture is opened
if not cap.isOpened():
    print("Error: Could not open video source.")
    exit()

# Directory to save temporary frames
temp_dir = "C:\\Users\\Elie_\\Desktop\\VS CODE PROJECTS\\Final Project Inmind\\Temp Frames"
os.makedirs(temp_dir, exist_ok=True)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Save the current frame to a temporary image file
    temp_image_path = os.path.join(temp_dir, 'current_frame.png')
    cv2.imwrite(temp_image_path, frame)

    # Change to the YOLOv7 directory and run the detect.py script on the current frame
    os.chdir(yolo_dir)
    os.system(f'python detect.py --weights runs/train/yolov7-w6-custom/weights/best.pt --img 640 --conf 0.25 --source "{temp_image_path}" --no-trace')

    # Load the processed image 
    output_dir = max([os.path.join(yolo_dir, 'runs', 'detect', d) for d in os.listdir(os.path.join(yolo_dir, 'runs', 'detect')) if 'exp' in d], key=os.path.getmtime)
    output_image_path = os.path.join(output_dir, 'current_frame.png')
    processed_frame = cv2.imread(output_image_path)

    # Display the processed frame
    if processed_frame is not None:
        cv2.imshow('Real-time YOLOv7 Inference', processed_frame)
    else:
        print("Error: Processed frame not found.")

    # Quit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()

# Remove temporary frames
import shutil
shutil.rmtree(temp_dir)
