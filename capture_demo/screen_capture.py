import cv2
import mss
import numpy as np


class ScreenCapture:
    def __init__(self):
        self.mCapture = mss.mss()
        self.mData = []
        self.mTop = 0
        self.mLeft = 0
        self.mWidth = 800
        self.mHeight = 640
    def setCaptureRange(self,top,left,width,height):
        self.mTop = top
        self.mLeft = 0
        self.mWidth = width
        self.mHeight = height
    #return a channel x witdh x height numpy array
    def getScreenNumpy(self,channel = 3):
        monitor = {"top": self.mTop, "left": self.mLeft, "width": self.mWidth, "height": self.mHeight}
        scr_img = np.array(self.mCapture.grab(monitor))
        self.mData = scr_img
        #post processing into numpy
        scr_img = np.moveaxis(scr_img,-1,0)
        scr_img = scr_img[0:channel,:,:]
        
        return scr_img
    def showCapture(self):
        cv2.imshow("OpenCV/Numpy normal", self.mData)
        cv2.waitKey(0)
if __name__ == "__main__":
    capture = ScreenCapture()
    array = capture.getScreenNumpy(4)
    print(array.shape)
