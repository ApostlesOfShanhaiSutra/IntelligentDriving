#import some useful packages
import matplotlib.pyplot as plt
import numpy as np
import cv2

def gray_image(image):
    return cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)

def gauss_blur(image,kernel_size):
    return cv2.GaussianBlur(image,(kernel_size,kernel_size),0)

def region_of_interests(image,vertices):
    mask = np.zeros_like(image)
    if len(vertices)>2:
        channel_count = image.shape[2]
        ignore_mask_color = (255,)*channel_count
    else:
        ignore_mask_color = 255
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    masked_image = cv2.bitwise_and(mask,image)
    return masked_image

def draw_lines(image,lines,color=[0, 255, 0], thickness=2):
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(image, (x1, y1), (x2, y2), color, thickness)

def hough_lines(image, rho, theta, threshold, min_line_length, max_line_gap):
    lines = cv2.HoughLinesP(image, rho, theta, threshold, np.array([]), minLineLength=min_line_length,
                            maxLineGap=max_line_gap)
    line_img = np.zeros((image.shape[0],image.shape[1],3),dtype=np.uint8)
    draw_lines(line_img,lines)
    return line_img

def line_detect(image):
    #change to gray picture
    gray = gray_image(image)
    cv2.imshow('gray',gray)

    #gaussian blur
    kernel_size = 9
    blur_gray = gauss_blur(image,kernel_size)
    cv2.imshow('blur', blur_gray)

    #canny edge detect
    low_threshold = 10
    high_threshold = 150
    edges = cv2.Canny(blur_gray,low_threshold,high_threshold)
    cv2.imshow('edges',edges)

    #ROI
    imshape = image.shape
    vertices = np.array([[(0,imshape[0]),(0.45*imshape[1],0.6*imshape[0]),(0.6*imshape[1],0.45*imshape[0]),(imshape[1],imshape[0])]],dtype=np.int32)
    masked_image = region_of_interests(edges,vertices)
    cv2.imshow('masked_image',masked_image)

    #hough detect
    rho = 1
    theta = np.pi / 180
    threshold = 100
    min_line_length = 150
    max_line_gap = 100
    line_image = hough_lines(masked_image, rho, theta, threshold, min_line_length, max_line_gap)
    cv2.imshow('hough_image',line_image)

    #add weight
    result = cv2.addWeighted(line_image,0.9,image,1.,0.)
    cv2.imshow('result',result)

#read an image
img = cv2.imread("test_images/input.jpg")
print('This image shape:',img.shape)

line_detect(img)


cv2.waitKey(10000)
cv2.destroyAllWindows()


