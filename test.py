import cv2
import numpy as np
import time


def box_detect(frame):
    H, W, _ = frame.shape
    X_mid_frame = W // 2 
    Y_mid_frame = H // 2 

    x_tol, y_tol = int(W * 0.05), int(H * 0.05)
    
    # HSV Processing for Red
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, np.array([0, 100, 100]), np.array([10, 255, 255]))
    mask2 = cv2.inRange(hsv, np.array([170, 100, 100]), np.array([180, 255, 255]))
    red_mask = mask1 + mask2

    # Cleaning up the noise
    kernel = np.ones((5, 5), np.uint8) 
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)

    # Finding Contours
    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    print(len(contours))
    for cnt in contours:
        # 1. Calculate the perimeter to determine epsilon
        perimeter = cv2.arcLength(cnt, True)
        # print(perimeter)
        # 2. Approximate the shape (low epsilon = more detail, high = simpler shape)
        # Usually 2% to 5% of the perimeter is a good threshold
        approx = cv2.approxPolyDP(cnt, 0.04 * perimeter, True)

        # 3. Check if the approximated shape has 4 vertices
        if len(approx) == 4:
            area = cv2.contourArea(cnt)
            if area > 500:  # Ignore small noise
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
                cv2.putText(frame, "Red Rectangle", (x, y - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # print(len(contours))
    # c_c=0
    # # 4. Loop through contours and draw bounding boxes
    # for cnt in contours:
    #     # Filter out small noise by checking the area
    #     if cv2.contourArea(cnt) > 0:
    #         c_c +=1
    #         x, y, w, h = cv2.boundingRect(cnt)
    #         cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    #         cv2.putText(frame, "Red Object", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    # print("totoal contours area > 500:",c_c)

    # display_text = "No target detected"
    # target_color = (0, 0, 255) # Red (BGR)

    # # Logic to find the largest rectangle
    # best_cnt = None
    # max_area = 0
    # for cnt in contours:
    #     area = cv2.contourArea(cnt)
    #     if area > 500:
    #         x, y, w, h = cv2.boundingRect(cnt)
    #         aspect_ratio = float(w) / h
    #         if 0.2 < aspect_ratio < 5.0 and area > max_area:
    #             best_cnt = cnt
    #             max_area = area

    # if best_cnt is not None:
    #     x, y, w, h = cv2.boundingRect(best_cnt)
    #     cx, cy = x + w//2, y + h//2
        
    #     # Check if centered
    #     if (X_mid_frame - x_tol < cx < X_mid_frame + x_tol) and \
    #         (Y_mid_frame - y_tol < cy < Y_mid_frame + y_tol):

    #         display_text = "Target Centered"
    #         target_color = (0, 255, 0) 
    #     else:
    #         # Direction Logic
    #         dir_x = "Left" if cx < X_mid_frame - x_tol else ("Right" if cx > X_mid_frame + x_tol else "")
    #         dir_y = "Up" if cy < Y_mid_frame - y_tol else ("Down" if cy > Y_mid_frame + y_tol else "")
    #         display_text = f"{dir_x} {dir_y}".strip()
    #         target_color = (255, 0, 0) 
    #     cv2.putText(frame, display_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    #     cv2.rectangle(frame, (x, y), (x + w, y + h), target_color, 3)
    #     cv2.circle(frame, (cx, cy), 5, (255, 255, 255), -1)
    #     # if(X_mid_frame - x_tol < cx < X_mid_frame + x_tol):

    # Draw Crosshair
    cv2.imshow('red_mask ',red_mask)
    cv2.imshow("frame",frame)
    cv2.waitKey(0)


if __name__ == "__main__":
    f = cv2.imread("photos/test2.jpg")
    f = cv2.resize(f, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

    # print(f)
    box_detect(f)
    input()