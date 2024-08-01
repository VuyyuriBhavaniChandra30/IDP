import cv2
import numpy as np
import os
import time
import torch
import torchvision.transforms as transforms
from PIL import Image,ImageOps
import subprocess
import shutil
from ultralytics import YOLO
def print_hyphens(s,n):
    print(s,end="<")
    for _ in range(n-2):
        print('-', end='', flush=True)  # Print the hyphen without a newline
        time.sleep(5 / n)  # Pause for 3 seconds divided by the length of hyphens
    print(">",end="")
    print()

def extractor(vid_path):
    frame_count=1
    cap = cv2.VideoCapture(vid_path)

    # Check if the video file was opened successfully
    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()

    # Initialize a list to store unique frames
    frames = []


    # Read the first frame
    ret, prev_frame = cap.read()
    print("Frame:",frame_count,end="")
    if ret:
        # Convert the frame to a grayscale image for faster comparison
        #prev_frame_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        frames.append(prev_frame)
        print("\tInserted")
    frame_count+=1
    while True:
        print("Frame:",frame_count,end="")
        ret, frame = cap.read()

        # Break the loop when we reach the end of the video
        if not ret:
            break

        # Convert the frame to grayscale for comparison
        #frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Calculate the absolute difference between the current and previous frames
        # frame_diff = cv2.absdiff(prev_frame_gray, frame_gray)
        frame_diff = cv2.absdiff(prev_frame, frame)
        # If the frame is not a duplicate (based on a simple threshold),
        # add it to the list of frames and update the previous frame
        if np.mean(frame_diff) > 5:
            frames.append(frame)
            print("\tInserted<-")
        #  prev_frame_gray = frame_gray.copy()
            prev_frame = frame.copy()
        else:
            print("\tRejected")
        frame_count+=1
        # Display the frame (optional)
        # cv2.imshow('Frame', frame_gray)
        # Press 'q' to exit the loop
        #if cv2.waitKey(1) & 0xFF == ord('q'):
        #    break

        # Release the video capture object and close any open windows
    cap.release()
    cv2.destroyAllWindows()

    video_name = os.path.basename(vid_path).split('.')[0]

    # frames_dict= { frame_name:frame 

    frames_dict={}
    
    for i, frame in enumerate(frames):
        frames_dict[video_name+"_frame_"+str(i)]=frame
    
    return frames_dict


def check_image_high_resolution(image):
    width, height = image.size
    if width*height > 89280:
        return True
    return False

def enhance_photos(input_path,photo_dict):
    
    extension=""
    if input_path[len(input_path)-3:]=="mp4":
        extension=".jpg"                                                                                                                                                                                                        
    #resolution Enhancer    
    lr_folder = 'C:\\Users\\vuyyu\\Desktop\\IDP\\ESRGAN\\LR'
    
    for photo_name, photo in photo_dict.items():
        
        if isinstance(photo, np.ndarray):
            input_image = Image.fromarray(cv2.cvtColor(photo, cv2.COLOR_BGR2RGB))  # Convert OpenCV frame to PIL Image
        else:
            input_image=photo
            
        print(photo_name, "loaded", end=" ---> ")

        if not check_image_high_resolution(input_image):
            print("uploaded")
            # If it's a valid image, save it to the LR folder
            photo_path = os.path.join(lr_folder, f'{photo_name}{extension}')
            cv2.imwrite(photo_path, photo)  # Saving the image with OpenCV (if needed)
        else:
            print("Failed ---> ",end="")
            photo_path = os.path.join(input_path, f'{photo_name}{extension}')
            
            input_image = ImageOps.grayscale(input_image)
            # Apply histogram equalization to enhance the image's contrast
            image_with_high_contrast = ImageOps.equalize(input_image)
            # Convert the PIL Image to a NumPy array
            image_np = np.array(image_with_high_contrast)
            cv2.imwrite(photo_path, image_np)
            print("Contrast Enhanced")
            
    
    print()
    print("\t------ Executing subprocess -------")
    print_hyphens("\t",len("------ Executing subprocess -------"))
    print()
    process = subprocess.Popen("cd C:\\Users\\vuyyu\\Desktop\\IDP\\ESRGAN && python test.py", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, error = process.communicate()
    return_code = process.returncode

    if return_code == 0:
        print("Subprocess executed. Transferring files...")
        source_path = "C:\\Users\\vuyyu\\Desktop\\IDP\\ESRGAN\\results"
        destination = './finalcode/input'

        for filename in os.listdir(source_path):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                
                new_filename = filename.replace("_rlt", "")
            
                file_path = os.path.join(source_path, filename)
                new_file_path = os.path.join(destination, new_filename)
                
                shutil.copy(file_path, new_file_path)
        
        print("Files transferred successfully.")
    
    if error:
        print("Error:", error.decode("utf-8"))
    print()
    print("\t------ 'Phase_2 : The Enhancer' completed -------")
    print()
    
    
input_path="./finalcode/vid1.mp4"
extension="" 
if input_path[len(input_path)-3:]=="mp4":
    extension=".jpg"
print()
print("\t------ 'Phase_1 : The Extraction' commenced -------")
print_hyphens("\t",len("------ 'Phase_1 : The Extraction' commenced -------"))
print()

ext_start=time.time()
ext_func_start=time.time()

if input_path[len(input_path)-3:]=="mp4":
    frames=extractor(input_path)
else:
    frames={}
    if os.path.exists(input_path):
        for filename in os.listdir(input_path):
            if filename.endswith(".jpg") or filename.endswith(".png"):  # Check for image files
                file_path = os.path.join(input_path, filename)
                frame = cv2.imread(file_path)
                frames[filename] = frame
ext_func_end=time.time()

print(frames.keys())

print()
print("\t------ Extraction completed -------")
time.sleep(1)
print()

#checking the frames
"""for frame_name, frame in frames.items():
    cv2.imshow(frame_name, frame)
    cv2.waitKey(0)  # Display each frame until any key is pressed
    cv2.destroyWindow(frame_name)  # Close the frame window after viewing

cv2.waitKey(0)  # Wait for any key to be pressed to close all frame windows
cv2.destroyAllWindows()"""

print()
print("\t------ Creating output Folder -------")
print_hyphens("\t",len("------ Creating output Folder -------"))
print()

ext_folder_creating_start=time.time()
output_folder = './finalcode/input'
if not os.path.exists(output_folder):
  os.makedirs(output_folder)
ext_folder_creating_end=time.time()

print()
print("\t------ Output Folder Created -------")
time.sleep(1)
print()
# Save the frames to the "input" folder

print()
print("\t------ Commencing File Upload -------")
print_hyphens("\t",len("------ Commencing File Upload -------"))
print()

ext_file_save_start=time.time()
video_name = os.path.basename(input_path).split('.')[0]

for i, frame in frames.items():
    print("Saving",i)
    frame_filename = os.path.join(output_folder, f'{i}{extension}')
    cv2.imwrite(frame_filename, frame)
ext_file_save_end=time.time()

ext_end=time.time()

print()
print("\t------ File Upload completed -------")
time.sleep(1)
print()

print()
print("\t------ 'Phase_1 : The Extraction' completed -------")
time.sleep(1)
print()

print()
print("\t------ 'Phase_2 : The Enhancer' commenced -------")
print_hyphens("\t",len("------ 'Phase_2 : The Enhancer' commenced -------"))
print()

#temp_frames= { "temp_photo":Image.open("./finalcode/comic.png") }

enh_func_start=time.time()
enhanced_frames=enhance_photos(input_path,frames)
enh_func_end=time.time()

"""count=0
for i, frame in frames.items():
  frame_filename = os.path.join(output_folder, f'{video_name}_frame_0{count}{extension}')
  cv2.imwrite(frame_filename, frame)
  count+=1
"""

#statistics
print("Extraction Total:",ext_end-ext_start-5)
print("\tExtraction Function:",ext_func_end-ext_func_start)
print("\tExtraction Folder Creation:",ext_folder_creating_end-ext_folder_creating_start)
print("\tExtraction File Uploading:",ext_file_save_end-ext_file_save_start)
print("Enhancer Total:",enh_func_end-enh_func_start)




# Load YOLO model
model = YOLO('yolov8n-seg.pt')

# Check if the input folder exists
if os.path.exists("C:\\Users\\vuyyu\\Desktop\\IDP\\finalcode\\input"):
    for filename in os.listdir("C:\\Users\\vuyyu\\Desktop\\IDP\\finalcode\\input"):
        # Check for image files
        if filename.endswith(".jpg") or filename.endswith(".png"):
            file_path = os.path.join("C:\\Users\\vuyyu\\Desktop\\IDP\\finalcode\\input", filename)

            # Check if the image file exists
            if os.path.isfile(file_path):
                # Load the image using Pillow
                img = Image.open(file_path)
                # Convert the image to OpenCV format
                img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                # Make predictions
                res = model(img)
                # Process the detection results
                for box in res[0].boxes:
                    class_name = res[0].names[int(box.cls)]
                    # Extract box coordinates
                    xyxy = box.xyxy[0].int().cpu().numpy()
                    x1, y1, x2, y2 = xyxy
                    # Extract segment from the image
                    segment = img[y1:y2, x1:x2]
                    # Display the segment with the class name
                    cv2.putText(segment, class_name, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
                    cv2.imshow('Segment', segment)
                    cv2.waitKey(0)  # Waits indefinitely until a key is pressed
