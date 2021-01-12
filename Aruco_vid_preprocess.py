import cv2
import numpy as np
import matplotlib.pyplot as plt
import pytesseract
import time


def createArucoData():
    aruco_dict = cv2.aruco.custom_dictionary(0, 4, 1)
    # add empty bytesList array to fill with 3 markers later
    aruco_dict.bytesList = np.empty(shape=(4, 2, 4), dtype=np.uint8)
    # add new markers
    mybits = np.array([[1, 0, 1, 1], [0, 1, 0, 1], [0, 0, 1, 1], [0, 0, 1, 0]], dtype=np.uint8)
    aruco_dict.bytesList[0] = cv2.aruco.Dictionary_getByteListFromBits(mybits)
    mybits = np.array([[0, 0, 0, 0], [1, 1, 1, 1], [1, 0, 0, 1], [1, 0, 1, 0]], dtype=np.uint8)
    aruco_dict.bytesList[1] = cv2.aruco.Dictionary_getByteListFromBits(mybits)
    mybits = np.array([[1, 0, 0, 1], [1, 0, 0, 1], [0, 1, 0, 0], [0, 1, 1, 0]], dtype=np.uint8)
    aruco_dict.bytesList[2] = cv2.aruco.Dictionary_getByteListFromBits(mybits)
    mybits = np.array([[0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 1, 0], [1, 1, 0, 1]], dtype=np.uint8)
    aruco_dict.bytesList[3] = cv2.aruco.Dictionary_getByteListFromBits(mybits)

    parameters = cv2.aruco.DetectorParameters_create()
    parameters.cornerRefinementMethod = 5
    parameters.errorCorrectionRate = 0.3

    return aruco_dict, parameters


def warpImage(image, srcPoint, dstPoint, shape):
    homoMat, status = cv2.findHomography(srcPoint, dstPoint)

    return cv2.warpPerspective(image, homoMat, shape)

def ROI_aruco(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    arucoDict, params = createArucoData()
    corners, ids, rejected = cv2.aruco.detectMarkers(gray, arucoDict, parameters=params)
    cv2.aruco.drawDetectedMarkers(gray, corners, ids)
    srcPoint = np.array([corners[3][0][3], corners[1][0][1], corners[2][0][2], corners[0][0][0]])
    dstPoint = np.array([[0, 0], [0, 1013], [784, 0], [784, 1013]])
    dstImage = warpImage(gray, srcPoint, dstPoint, (784, 1013))
    return dstImage

def remove_shadow(img):
    rgb_planes = cv2.split(img)
    result_planes = []
    result_norm_planes = []
    for plane in rgb_planes:
        dilated_img = cv2.dilate(plane, np.ones((7,7), np.uint8))
        bg_img = cv2.medianBlur(dilated_img, 21)
        diff_img = 255 - cv2.absdiff(plane, bg_img)
        norm_img = cv2.normalize(diff_img,None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        result_planes.append(diff_img)
        result_norm_planes.append(norm_img)
    result = cv2.merge(result_planes)
    result_norm = cv2.merge(result_norm_planes)
#     return result_norm
    return result

def extract(img):
    # gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    threshold_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    config='-l eng + equ'
    details = pytesseract.image_to_string(threshold_img,config=config)
    return details

def camera_interact(vid_path):
    cap = cv2.VideoCapture(vid_path)
    content="-1"
    count=0
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            count+=1
            frame= preprocess(frame)
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            if count %10 ==0:
                content = extract(frame)
            if count == 200:
                break
        else:
            break

    # Release everything if job is finished
    cap.release()
    cv2.destroyAllWindows()
    return content

def preprocess(img):
    img=ROI_aruco(img)
    img=remove_shadow(img)
    img=cv2.resize(img,(600,600))
    # return img,content
    return img

cont=camera_interact("./20210112_143005000_iOS.avi")
print(cont)