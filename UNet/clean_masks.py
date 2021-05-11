import cv2
import os
from matplotlib import pyplot as plot

#MASKS_PATH = os.path.abspath(os.path.join(os.getcwd(), "Dataset", "train", "lung_aca", "masks", "data"))
MASKS_PATH = os.path.abspath(os.path.join(os.getcwd(), "Dataset", "test", "lung_aca", "masks", "data"))

ctr = 0
for img_name in os.listdir(MASKS_PATH):
	img = cv2.imread(os.path.join(MASKS_PATH, img_name), cv2.IMREAD_GRAYSCALE)
	img[img<127] = 0
	img[img>128] = 255
	cv2.imwrite(os.path.join(MASKS_PATH, img_name), img)
	ctr += 1
	if(ctr%50==0):
		print(ctr)