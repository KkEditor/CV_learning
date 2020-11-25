import cv2
import numpy as np
from matplotlib import pyplot as plt
#Cac ki thuat tien xu ly se ap dung

#Phan tich du lieu:
# Nhieu anh bi vuong toc -> hair removal
# Nhieu anh co do tuong ban khong noi bat -> Histogram Equalize
# Anh co kich co khong giong nhau -> image resize
# Lam smooth anh -> Noise reduciton
#Qua trinh:
#Input image ->Preprocessing
#Khoanh vung doi tuong -> ROI
#Ap dung ROI vao Input Image -> Input cho ML
#xoa muc


 #restore from blur


def preprocess(img):
    img=hair_removal(img)
    img=remove_ink(img)
    mask=ROI(img)
    res=apply_mask(img,mask)
    return res

def ROI(img):
    gr=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    mean,std= cv2.meanStdDev(gr)
    ret, thresh2 = cv2.threshold(gr, int(mean-std), 255, cv2.THRESH_BINARY_INV)
    return thresh2

def apply_mask(img,mask):
    res = cv2.bitwise_and(img, img, mask=mask)
    return res

def resize(img,size):
    return cv2.resize(img,(size,size))

def hair_removal(img):
    grayScale = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)   # chuyen ve anh xam
    kernel = cv2.getStructuringElement(1, (17, 17))     # khoi tao cua so truot
    blackhat = cv2.morphologyEx(grayScale, cv2.MORPH_BLACKHAT, kernel)   # su khac nhau giu dilation+erosion voi anh goc
    ret, thresh2 = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)   # mask de remove hair
    # thresh2=cv2.equalizeHist(thresh2)   #lam tang do tuong phan
    dst = cv2.inpaint(img, thresh2, 1, cv2.INPAINT_TELEA)   #ghep mask voi anh goc
    # cv2.imshow("thresh", thresh2)
    # cv2.imshow("blackhat",blackhat)
    return dst

def remove_ink(img):
    hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    lower_purple = np.array([40, 70, 70])
    upper_purple = np.array([180, 255, 255])
    mask = cv2.inRange(hsv, lower_purple, upper_purple)
    res = cv2.bitwise_and(img, img, mask=mask)
    cv2.imshow("m",res)
    ret, thresh2 = cv2.threshold(res, 100, 255, cv2.THRESH_BINARY)
    thresh2=cv2.cvtColor(thresh2,cv2.COLOR_BGR2GRAY)
    dst=cv2.bitwise_not(img,img,mask=thresh2)
    return dst

def main():
    img = cv2.imread("F:/jupyter_notebook/data/siim-isic-melanoma-classification/jpeg/train/ISIC_0164329.jpg")
    img=resize(img,700)
    res=preprocess(img)
    cv2.imshow("origin",img)
    cv2.imshow("result",res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



if __name__ == '__main__':
    main()