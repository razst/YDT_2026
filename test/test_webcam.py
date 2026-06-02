import cv2
import sys

def check_camera(device_index=0):
    # Initialize VideoCapture
    cap = cv2.VideoCapture(device_index)

    # Check if camera opened successfully
    if not cap.isOpened():
        print(f"Error: Could not open camera at index {device_index}")
        return False

    # Attempt to read a single frame to verify the stream
    ret, frame = cap.read()

    if not ret or frame is None:
        print("Error: Camera opened but failed to capture a frame.")
        cap.release()
        return False

    print(f"Success: Camera at index {device_index} is connected and streaming.")
    
    # Clean up
    cap.release()
    return True

if __name__ == "__main__":
    # Check for index 0 (default USB camera)
    if not check_camera(0):
        sys.exit(1)