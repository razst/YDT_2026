import cv2
import numpy as np

# Video sources
video_src = "C:/dev/GitHub/PiCam_20260604_190530.avi"
cap = cv2.VideoCapture(1)

hsv_values = {
    "hMin": 0, "hMax": 179,
    "sMin": 0, "sMax": 255,
    "vMin": 0, "vMax": 255,
}

current_mode = "custom"
is_updating_sliders = False  # Prevents code-updates from breaking preset mode

def on_trackbar(value, key):
    global current_mode
    hsv_values[key] = value
    # If the user manually drags the Hue sliders, disable preset mode
    if not is_updating_sliders and key in ["hMin", "hMax"]:
        current_mode = "custom"

def set_hsv_ranges(h_min, h_max, s_min, s_max, v_min, v_max):
    global is_updating_sliders
    is_updating_sliders = True
    cv2.setTrackbarPos("H Min", "HSV Calibration", h_min)
    cv2.setTrackbarPos("H Max", "HSV Calibration", h_max)
    cv2.setTrackbarPos("S Min", "HSV Calibration", s_min)
    cv2.setTrackbarPos("S Max", "HSV Calibration", s_max)
    cv2.setTrackbarPos("V Min", "HSV Calibration", v_min)
    cv2.setTrackbarPos("V Max", "HSV Calibration", v_max)
    is_updating_sliders = False

def handle_calibration_clicks(event, x, y, flags, param):
    global current_mode
    if event == cv2.EVENT_LBUTTONDOWN:
        # Check if clicks fall within our drawn UI buttons
        if 10 <= y <= 50:
            if 10 <= x <= 110:
                print("Red 1 Selected (Lower: 0-10)")
                current_mode = "red1"
                set_hsv_ranges(0, 10, 100, 255, 100, 255)
            elif 120 <= x <= 220:
                print("Red 2 Selected (Upper: 160-179)")
                current_mode = "red2"
                set_hsv_ranges(160, 179, 100, 255, 100, 255)
            elif 230 <= x <= 330:
                print("Green Preset Selected")
                current_mode = "green"
                set_hsv_ranges(40, 80, 100, 255, 100, 255)
            elif 340 <= x <= 440:
                print("Blue Preset Selected")
                current_mode = "blue"
                set_hsv_ranges(100, 140, 100, 255, 100, 255)
            elif 450 <= x <= 550:
                print("Yellow Preset Selected")
                current_mode = "yellow"
                set_hsv_ranges(15, 35, 60, 255, 90, 255)

def pick_color(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        try:
            pixel = hsv[y, x]
            print(f'HSV at ({x}, {y}): {pixel}')
        except NameError:
            pass

# Initialize Windows
cv2.namedWindow("HSV Calibration")
cv2.namedWindow("HSV Mask")
cv2.namedWindow("Image")

# Attach mouse callbacks to their respective windows
cv2.setMouseCallback("HSV Calibration", handle_calibration_clicks)
cv2.setMouseCallback("Image", pick_color)

# Create Trackbars on the Calibration Window
cv2.createTrackbar("H Min", "HSV Calibration", 0, 179, lambda value: on_trackbar(value, "hMin"))
cv2.createTrackbar("H Max", "HSV Calibration", 179, 179, lambda value: on_trackbar(value, "hMax"))
cv2.createTrackbar("S Min", "HSV Calibration", 0, 255, lambda value: on_trackbar(value, "sMin"))
cv2.createTrackbar("S Max", "HSV Calibration", 255, 255, lambda value: on_trackbar(value, "sMax"))
cv2.createTrackbar("V Min", "HSV Calibration", 0, 255, lambda value: on_trackbar(value, "vMin"))
cv2.createTrackbar("V Max", "HSV Calibration", 255, 255, lambda value: on_trackbar(value, "vMax"))

kernel = np.ones((5,5), np.uint8)

# Expanded the width from 350 to 460 to fit the fourth button, and to 560 for yellow
calibration_panel = np.zeros((70, 560, 3), dtype=np.uint8)

# Button 1: RED 1 (Lower HSV)
cv2.rectangle(calibration_panel, (10, 10), (110, 50), (0, 0, 150), -1) # Slightly darker red
cv2.putText(calibration_panel, "RED 1", (30, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

# Button 2: RED 2 (Upper HSV)
cv2.rectangle(calibration_panel, (120, 10), (220, 50), (0, 0, 255), -1) # Brighter red
cv2.putText(calibration_panel, "RED 2", (140, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

# Button 3: GREEN
cv2.rectangle(calibration_panel, (230, 10), (330, 50), (0, 255, 0), -1)
cv2.putText(calibration_panel, "GREEN", (245, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

# Button 4: BLUE
cv2.rectangle(calibration_panel, (340, 10), (440, 50), (255, 0, 0), -1)
cv2.putText(calibration_panel, "BLUE", (365, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

# Button 5: YELLOW
cv2.rectangle(calibration_panel, (450, 10), (550, 50), (0, 255, 255), -1)
cv2.putText(calibration_panel, "YELLOW", (465, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


while True:
    success, image = cap.read()
    if not success:
        cap = cv2.VideoCapture(video_src)
        success, image = cap.read()

    cv2.imshow("Image", image)
    
    # Display the button panel in the Calibration window above the trackbars
    cv2.imshow("HSV Calibration", calibration_panel)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Fetch current slider values for S and V
    s_min, s_max = hsv_values["sMin"], hsv_values["sMax"]
    v_min, v_max = hsv_values["vMin"], hsv_values["vMax"]
    
    # --- MASKING LOGIC ---
    # Because you are calibrating RED 1 and RED 2 individually, we can just 
    # use a standard contiguous mask based strictly on the slider values.
    lower_bound = np.array([hsv_values["hMin"], s_min, v_min])
    upper_bound = np.array([hsv_values["hMax"], s_max, v_max])
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    
    # Clean up the mask
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    cv2.imshow("HSV Mask", mask)

    # Wait for spacebar to advance or ESC to exit
    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == 27:  # ESC to exit
            break
        elif key == 32:  # SPACE to advance
            break
    
    if key == 27:  # If ESC was pressed, exit outer loop
        break

cv2.destroyAllWindows()