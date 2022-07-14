import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread('lung_n.jpeg', cv2.IMREAD_UNCHANGED)
plt.imshow(img)

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img_rgb)

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, binary = cv2.threshold(img_gray, 120, 255, cv2.THRESH_BINARY_INV)
plt.imshow(binary, cmap="gray")
plt.show()

ret, thresh = cv2.threshold(img_gray, 120, 255, 0)
_, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
"""
i = 1
cnt = contours[i]
cv2.drawContours(img, [cnt], 0,(255,0,0), 0)
print("Showing {}th contour".format(i+1))
plt.imshow(img)
plt.show()
"""

cnts = contours

print(len(contours))
cv2.drawContours(img, contours, 0, 255, -1)
for cnt in cnts:
    area = cv2.contourArea(cnt)
    print(area)
    x,y,w,h = cv2.boundingRect(cnt) # offsets - with this you get 'mask'
    #cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
    print('Average color (BGR): ',np.array(cv2.mean(img[y:y+h,x:x+w])).astype(np.uint8))
plt.imshow(img)
plt.show()

# CHANGES

# Area threshold of 800 is too high. Performing
