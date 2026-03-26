import cv2
import numpy as np
from enum import Enum
import time
class Detect:
    def __init__(self,frame_queue,auto_start = False):
        self.lower_red_1 = np.array([0, 80, 50])    # Lowered Saturation and Value minimums
        self.upper_red_1 = np.array([20, 255, 255])
        self.lower_red_2 = np.array([160, 80, 50])
        self.upper_red_2 = np.array([180, 255, 255])
        self.CENTER_THRESHOLD = 0.25
        self.frame_queue = frame_queue
    #BUG what happens if we see two H lines?

    # RED color mask
    # Define a WIDER red color range in HSV
    # This accounts for lighting variations and different shades of red tape.
    # TODO to cnost
    def sort_rect(rect):
        return(rect[0])

    class TargetPosition(Enum):
        LEFT = 0
        RIGHT = 1
        CENTER = 2
        CENTER_FIRE = 3
        NOT_DETECTED = 4 # we didn't detect two H lines ! (need to go up)

    
    def detect_x_lines(self) ->TargetPosition:                
        """
        Detects red lines in an image using a wider HSV range and robust filtering.
        Saves the processed mask and the final detected image.
        """

        global center_frame
        try:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask1 = cv2.inRange(hsv, self.lower_red_1, self.upper_red_1)
            mask2 = cv2.inRange(hsv, self.lower_red_2, self.upper_red_2)
            red_mask = mask1 + mask2
            _, red_mask = cv2.threshold(red_mask, 150, 255, cv2.THRESH_BINARY)
            kernel_height = 20
            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_height))


            red_mask = cv2.erode(red_mask, horizontal_kernel, iterations=2)  # More iterations
            red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, horizontal_kernel)  # Morphological closing
            red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, horizontal_kernel)  # Morphological opening
            cv2.imshow("red_vertical",red_mask)
            contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            image_height, image_width, _ = frame.shape
            min_line_area = image_height * image_width * 0.001 #bigger then half a % of the screen 
            rect_locations = []
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            # Process the two largest contours, if they meet the area threshold
            for i, contour in enumerate(contours): # Look at most 2 largest
                area = cv2.contourArea(contour)
                if area > min_line_area:
                    x, y, w, h = cv2.boundingRect(contour)
                    rect = (x,y,x+w,y+h)
                    if h > 30: #bigger then 1% of the screen  #TODO param in const ???                     
                        rect_locations.append(rect)
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (255,0,255), 2)
            # TODO return NOT DETECTED if num of lines < 2
            if len(rect_locations) < 2:
                return TargetPosition.NOT_DETECTED 
            rect_locations.sort(key=sort_rect)
            #rect_locatins((x1,y1,xw1,yh1),(x2,y2,xw2,yh2))
            print(rect_locations) 
            right_w = abs(image_width-rect_locations[1][0]-center_frame[1])
            left_w = abs(center_frame[1] - rect_locations[0][2]) 
            w = rect_locations[1][0]-rect_locations[0][2] 
            print(w,CENTER_THRESHOLD * w)

            if abs(left_w-right_w)<CENTER_THRESHOLD * w:
                if (center_frame[0] <= rect_locations[0][3]) and (rect_locations[0][1] <= center_frame[0]):
                    print("center_fire")
                    return TargetPosition.CENTER_FIRE
                print("center")
                return TargetPosition.CENTER
            elif left_w>=right_w:                
                print("left")
                return TargetPosition.LEFT
            elif right_w > left_w:
                print("right")
                return TargetPosition.RIGHT  


        except Exception as e:
            print(f"An error occurred while processing frame: {e}")
            return []

    def show_image(frame,fps):

        cv2.putText(frame, f"FPS: {fps:.2f}", (10,150),
        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        #TODO make this the only detect that is needed

        position = detect_x_lines(frame)
        put_text = ''
        if position == TargetPosition.CENTER_FIRE:
            put_text = f"center fire"
        if position == TargetPosition.LEFT:
            put_text = f"go left"
        if position == TargetPosition.CENTER:
            put_text="center"
        if position == TargetPosition.RIGHT:
            put_text =  f"go right"
        if position == TargetPosition.NOT_DETECTED:
                put_text = f"not detected"
        cv2.putText(frame, put_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        radius_circle=20
        thickness_circle=2
        color_circle=(0, 0, 255)
        circle_x = center_frame[1]
        circle_y = center_frame[0]
        center_circle=(circle_x,circle_y)
        cv2.line(frame, (center_frame[1]-radius_circle, center_frame[0]), (circle_x+radius_circle, circle_y), color_circle, 2)
        cv2.line(frame, (circle_x, circle_y-radius_circle), (circle_x, circle_y+radius_circle), color_circle, 2)
        cv2.imshow("final",frame)
    def start(self):
        thread = threading.Thread(target=self.update, daemon=True) 
        thread.start()
        return self
    def main():
        # --- Main Execution ---
        # Path to your video file
        video_path = "C:/Users/user/Downloads/field_test2.mp4" # test diffremt highths 
        # video_path = "C:/Users/user/Downloads/field_test1.mp4" # test all positions
        cap = cv2.VideoCapture(video_path)
        prev_time = 0
        fps_count=0
        fps_sum=0

        if not cap.isOpened():
            print("Error: Could not open video.")
            exit()
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            current_time = time.time()

            fps = 1 / (current_time - prev_time) if prev_time != 0 else 0
            prev_time = current_time
            fps_count +=1
            frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA) 
            height,width = frame.shape[:2]
            global center_frame
            center_frame = (height//2,width//2)
            show_image(frame,fps)

            # Wait for ANY key to go to next frame
            key = cv2.waitKey(0)  # 0 = wait forever
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
    main()