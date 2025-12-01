import cv2
import numpy as np

img = cv2.imread("photos\WIN_20251127_19_27_09_Pro.jpg")

columns = np.array_split(img, 3, axis=1)
for i, col in enumerate(columns):
    cv2.imwrite(f"col_{i}.jpg", col)

rows = np.array_split(img, 6, axis=0)
for i, row in enumerate(rows):
    cv2.imwrite(f"row_{i}.jpg", row)

small = cv2.resize(img, None, fx=0.4, fy=0.4, interpolation=cv2.INTER_AREA)
cv2.imshow("image",small)
cv2.waitKey(0)
