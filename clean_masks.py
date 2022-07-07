import cv2
import os

#topDir='./unet_trainset/lung_aca/'
topDir='./Images/Trainset/lung_aca/'

ctr = 0
for path in os.listdir(topDir+'masks/'):
    img = cv2.imread(topDir+'masks/'+path, 0)
    img[img<=127] = 0
    img[img>127] = 1
    cv2.imwrite(topDir+'masks_cleaned/'+path, img)
    ctr += 1
    if(ctr%50==0):
        print(ctr, "done")
