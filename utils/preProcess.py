import cv2
from PIL import Image, ImageChops
import matplotlib.pyplot as plt
import os

def diff(img, img1):  # returns just the difference of the two images
    return cv2.absdiff(img, img1)


def preProcess(img0,img1):
    diff = ImageChops.subtract(img0, img1)
    diff = diff.point(lambda i: i * 5)
    w, h = diff.size
    area = (9, 70, 115, 229)
    diff = diff.crop(area)
    return diff


#img1 = Image.open('/home/jasper/Documents/BP_Jasp/data/pg/all/T0412_S008_U014/444-1.jpg' )
#img2 = Image.open('/home/jasper/Documents/BP_Jasp/data/pg/all/T0412_S008_U014/444-2300.jpg' )


#img1=img1.convert('LA')
#img2=img2.convert('LA')
#diff = ImageChops.subtract(img1, img2)
#diff = diff.point(lambda i: i * 5)
#w ,h = diff.size
#area = (9,70,115,229)
#diff.crop(area).show()
