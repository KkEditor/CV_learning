import cv2
import numpy as np


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

def preprocess(img):
    img=hair_removal(img)
    # img=hist_equalize(img)
    # img=noise_reduction(img,5)
    # img =cv2.equalizeHist(img)
    return img

def noise_reduction(img,kernel):
    img=cv2.GaussianBlur(img,(kernel,kernel),sigmaX=10)
    return img
def hist_equalize(img):
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    # equalize the histogram of the Y channel
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
    # convert the YUV image back to RGB format
    img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    return img_output


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
def main():
    img = cv2.imread("F:/jupyter_notebook/data/siim-isic-melanoma-classification/jpeg/train/ISIC_0109703.jpg")
    img=resize(img,700)
    res=preprocess(img)
    cv2.imshow("origin",img)
    cv2.imshow("result",res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



if __name__ == '__main__':
    main()