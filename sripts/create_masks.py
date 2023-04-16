import numpy as np
import cv2
from pynput import keyboard
import os
from matplotlib import pyplot as plot

BASE_PATH = os.getcwd()
RESULT_DIR = "Result"
DATA_DIR = "Dataset"
DATA_SUBDIRS = ["lung_aca", "lung_n", "lung_scc"]
DATA_PATHS = list(map(lambda x:os.path.join(BASE_PATH, DATA_DIR, x), DATA_SUBDIRS))
RESULT_PATHS = list(map(lambda x:os.path.join(BASE_PATH, RESULT_DIR, x), DATA_SUBDIRS))

def add_to_filename(path, cont):
    name, extension = tuple(path.split('.'))
    return ".".join((name+cont, extension))

def draw_contours(image, contours, colors=[(0,0,255),(255,0,0),(0,255,0)]):
    num_colors = len(colors)
    for i in range(len(contours)):
        cv2.drawContours(image, [contours[i]], -1, colors[i%num_colors], 3)


def find_contour(contours, coordinate):
    # Change to find innermost contour
    for i in range(len(contours)):
        if(cv2.pointPolygonTest(contours[i], coordinate, False)!=-1): # Not outside
            return i
    return None


def register_contour(event, reqd_cntrs_idx):
    if contours is not None:
        cnt_idx = find_contour(contours, (event.xdata, event.ydata))
        if cnt_idx is None:
            print("Enclosing contour NOT FOUND")
        else:
            reqd_cntrs_idx.add(cnt_idx)
            print("Enclosing contour REGISTERED")
    else:
        print("No contours to check")

for data_path, result_path in zip(DATA_PATHS, RESULT_PATHS):

    print()
    image_paths = list(os.listdir(data_path))
    contours = None

    for img_path in image_paths:

        reqd_cntrs_idx = set()
        current_pic = os.path.join(data_path, img_path)
        img = cv2.imread(current_pic)
        if img is None:
            print("Image not read")
            print(img_path)
            continue

        imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(imgray,127, 255, cv2.THRESH_OTSU)
        _, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)

        # Plot the found contours
        draw_contours(img, contours)

        fig = plot.figure()
        axis = fig.add_subplot(111)
        axis.imshow(img)
        canvas_id = fig.canvas.mpl_connect('button_press_event', lambda event: register_contour(event, reqd_cntrs_idx))
        plot.show()

        mask_img_white = np.zeros_like(img)
        mask_img_data = np.zeros_like(img)

        # Draw Contours
        reqd_cntrs = np.array(contours, dtype='object')[list(reqd_cntrs_idx)]
        draw_contours(mask_img_white, reqd_cntrs, [255])
        draw_contours(mask_img_data, reqd_cntrs, [(255,255,255)])

        # Fill with data or white
        cv2.fillPoly(mask_img_white, reqd_cntrs, (255,255,255))
        mask_img_data[mask_img_white==255] = img[mask_img_white==255]

        #fig = plot.figure()
        #axis = fig.add_subplot(111)
        #axis.imshow(mask_img_data)
        #plot.show()

        print(result_path)
        print("Saved") if (cv2.imwrite(os.path.join(result_path, add_to_filename(img_path, '_white')), mask_img_white)) else print("Error Saving Image")
        print("Saved") if (cv2.imwrite(os.path.join(result_path, add_to_filename(img_path, '_data')), mask_img_data)) else print("Error Saving Image")

"""
# SHOWS ALL THE SELECTED CONTOURS
draw_contours(input_image, np.array(contours, dtype='object')[list(reqd_cntrs[0])])
print(np.array(contours, dtype='object')[list(reqd_cntrs[0])])
fig = plot.figure()
axis = fig.add_subplot(111)
axis.imshow(input_image)
plot.show()
"""
