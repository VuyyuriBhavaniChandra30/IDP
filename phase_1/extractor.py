import cv2
import numpy as np
import os
import time

# Open the video file
video_path = './phase_1/vid2.mp4'
cap = cv2.VideoCapture(video_path)

# Check if the video file was opened successfully
if not cap.isOpened():
  print("Error: Could not open video.")
  exit()

# Initialize a list to store unique frames
frames = []

#timer
start=time.time()

# Read the first frame
ret, prev_frame = cap.read()

# Convert the frame to a grayscale image for faster comparison
#prev_frame_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

frames.append(prev_frame)

while True:
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
  #  prev_frame_gray = frame_gray.copy()
    prev_frame = frame.copy()
      
 # Display the frame (optional)
 # cv2.imshow('Frame', frame_gray)
 # Press 'q' to exit the loop
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

# Release the video capture object and close any open windows
cap.release()
cv2.destroyAllWindows()
end=time.time()
# Create a folder called "input" if it doesn't exist
output_folder = './phase_1/input'
if not os.path.exists(output_folder):
  os.makedirs(output_folder)

# Save the frames to the "input" folder

video_name = os.path.basename(video_path).split('.')[0]

for i, frame in enumerate(frames):
  frame_filename = os.path.join(output_folder, f'{video_name}_frame_{i}.jpg')
  cv2.imwrite(frame_filename, frame)

print(f"Saved {len(frames)} frames without duplicates in the 'input' folder.")
print("The total time elapsed:",end-start)