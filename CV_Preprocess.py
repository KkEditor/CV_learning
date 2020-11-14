import cv2
import numpy as np
from PIL import Image



#ROI
#Hair removal
#Noise reduction
#resize
#histogram equalize


def preprocess(img):
    img =resize(img,700)
    img=hair_removal(img)

    # img=noise_reduction(img,5)
    # img =cv2.equalizeHist(img)
    return img

def noise_reduction(img,kernel):
    img=cv2.GaussianBlur(img,(kernel,kernel),sigmaX=10)
    return img

def resize(img,size):
    return cv2.resize(img,(size,size))

def hair_removal(img):
    grayScale = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    kernel = cv2.getStructuringElement(1, (15, 15))
    blackhat = cv2.morphologyEx(grayScale, cv2.MORPH_BLACKHAT, kernel)
    ret, thresh2 = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    thresh2=cv2.equalizeHist(thresh2)
    dst = cv2.inpaint(img, thresh2, 1, cv2.INPAINT_TELEA)
    return dst
def main():
    img = cv2.imread("F:/jupyter_notebook/data/siim-isic-melanoma-classification/jpeg/train/ISIC_0206442.jpg")
    res=preprocess(img)
    img=cv2.resize(img,(700,700))
    cv2.imshow("origin",img)
    cv2.imshow("result",res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



if __name__ == '__main__':
    main()