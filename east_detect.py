from imutils.object_detection import non_max_suppression
import numpy as np
import cv2
import keras
import pandas as pd
from PIL import Image

#khai báo nhãn
class_name_text = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 
'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R',
'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 
'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 
's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

#load model nhận dạng
load_model = keras.models.load_model('./modelkeras_conv.h5')
#load model phát hiện
east = "frozen_east_text_detection.pb"
#ham detect van ban
# def detect_east(img):
#     #tien xu ly
#     orig = img.copy()
#     (H,W) = img.shape[:2]
#     (newW, newH) = (320,320)
#     rW = W / float(newW)
#     rH = H / float(newH)
#     newimg = cv2.resize(img, (newW, newH))
#     (H, W) = newimg.shape[:2]
#     #khai bao dau ra
#     layerNames = ["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"]
#     #chuyen tiep qua mang
#     net = cv2.dnn.readNet(east)
#     blob = cv2.dnn.blobFromImage(img, 1.0,(W,H),(123.68, 116.78, 103.94), swapRB=True, crop=False)
#     net.setInput(blob)
#     (score, geometry) = net.forward(layerNames)
#     (numRows, numCols) = score.shape[2:4]
#     rects=[]
#     confidences = []
#     for y in range(0,numRows):
#         scoresData = score[0,0,y]
#         xData0 = geometry[0,0,y]
#         xData1 = geometry[0,1,y]
#         xData2 = geometry[0,2,y]
#         xData3 = geometry[0,3,y]
#         anglesData = geometry[0,4,y]
#         for x in range(0, numCols):
#             if scoresData[x] < 0.5:
#                 continue
#             (offsetX, offsetY) = (x * 4.0, y * 4.0)
#             #tinh chieu rong va chieu cao cua hop gioi han
#             angle = anglesData[x]
#             cos = np.cos(angle)
#             sin = np.sin(angle)
#             h = xData0[x] + xData2[x]
#             w = xData1[x] + xData3[x]
#             #tim toa do cua hop
#             endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
#             endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
#             startX = int(endX - w)
#             startY = int(endY - h)
#             #cho vao mang da dc khoi tao
#             rects.append((startX, startY, endX, endY))
#             confidences.append(scoresData[x])
#     boxes = non_max_suppression(np.array(rects), probs=confidences)
#     cut_arr = []
#     arr_x =[]
#     column = ['x','y']
#     for box in boxes:
#         td_arr = []
#         return_arr = []
#         startX = int(box[0] * rW)
#         startY = int(box[1] * rH)
#         endX = int(box[2] * rW)
#         endY = int(box[3] * rH)
#         # cv2.rectangle(img,(startX, startY), (endX, endY), (0, 255, 0), 2)
#         cut = img[startY:endY, startX:endX]
#         cut_arr.append(cut)
#         arr_x.append([startX, startY])
#         df = pd.DataFrame(arr_x, columns=column)
#         df_x = df.sort_values('x',ascending = True)
#         df_y = df_x.sort_values('y', ascending = True)
#         df_y = df_y.reset_index()
#         for n in range (len(df_y)):
#             td_arr.append(df.iloc[n,0])
#         for td in list(td_arr):
#             return_arr.append(cut_arr[td])
#     return img, return_arr
def detect_east(img):
    #tien xu ly
    orig = img.copy()
    (H,W) = img.shape[:2]
    (newW, newH) = (320,320)
    rW = W / float(newW)
    rH = H / float(newH)
    newimg = cv2.resize(img, (newW, newH))
    (H, W) = newimg.shape[:2]
    #khai bao dau ra
    layerNames = ["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"]
    #chuyen tiep qua mang
    net = cv2.dnn.readNet(east)
    blob = cv2.dnn.blobFromImage(img, 1.0,(W,H),(123.68, 116.78, 103.94), swapRB=True, crop=False)
    net.setInput(blob)
    (score, geometry) = net.forward(layerNames)
    (numRows, numCols) = score.shape[2:4]
    rects=[]
    confidences = []
    for y in range(0,numRows):
        scoresData = score[0,0,y]
        xData0 = geometry[0,0,y]
        xData1 = geometry[0,1,y]
        xData2 = geometry[0,2,y]
        xData3 = geometry[0,3,y]
        anglesData = geometry[0,4,y]
        for x in range(0, numCols):
            if scoresData[x] < 0.5:
                continue
            (offsetX, offsetY) = (x * 4.0, y * 4.0)
            #tinh chieu rong va chieu cao cua hop gioi han
            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)
            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]
            #tim toa do cua hop
            endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
            endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
            startX = int(endX - w)
            startY = int(endY - h)
            #cho vao mang da dc khoi tao
            rects.append((startX, startY, endX, endY))
            confidences.append(scoresData[x])
    boxes = non_max_suppression(np.array(rects), probs=confidences)
    cut_arr = []
    arr_x =[]
    column = ['x','y']
    for box in boxes:
        startX = int(box[0] * rW)
        startY = int(box[1] * rH)
        endX = int(box[2] * rW)
        endY = int(box[3] * rH)
        print("box",box)
        # cv2.rectangle(img,(startX, startY), (endX, endY), (0, 255, 0), 2)
        cut = img[startY:endY, startX:endX]
        td_arr = []
        return_arr = []
        cut_arr.append(cut)
        # print(cut_arr)
        arr_x.append([startX, startY])
        # # print ("cut",cut_arr)
        # print("x", arr_x)
        df = pd.DataFrame(arr_x, columns=column)
        # print ("df",df)
        df_x = df.sort_values('x',ascending = True)
        # print("df_x",df_x)
        df_y = df_x.sort_values('y', ascending = True)
        # print(df_y)
        df_y = df_y.reset_index()
        # print(df_y)
        for n in range (len(df_y)):
            td_arr.append(df_y.iloc[n,0])
        # print(df_y.iloc[3])
        # print("td_arr",td_arr)
        # print (len(df_y))
        for td in list(td_arr):
            return_arr.append(cut_arr[td])
            print (return_arr)
    return img, return_arr
#ham xu ly anh
def pre_pro(img):
    columns = ['x']
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)#chuyen anh xam
    noise = cv2.fastNlMeansDenoising(img,None,15,7,21)# khu nhieu
    cala = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8,8))
    contrast = cala.apply(noise)#tang tuong phan
    blur = cv2.blur(contrast,(7,3))#lam mo
    #nhi phan lan 1
    # thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,17,5)
    thresh_INV = cv2.threshold(contrast,90,255,cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)[1]
    kernel_INV = np.ones((3,2),np.uint8)
    kernel2_INV = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,2))
    ero_INV = cv2.erode(thresh_INV,kernel_INV,iterations=1)
    dilate_INV = cv2.dilate(ero_INV, kernel2_INV, iterations=1)
    contours_INV, hierachy_INV = cv2.findContours(thresh_INV,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    #nhi phan lan 2
    thresh = cv2.threshold(contrast,90,255,cv2.THRESH_BINARY|cv2.THRESH_OTSU)[1]
    kernel = np.ones((3,2),np.uint8)
    kernel2 = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,2))
    ero = cv2.erode(thresh,kernel,iterations=1)
    dilate = cv2.dilate(ero, kernel2, iterations=1)
    contours, hierachy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    # # # print(contours)
    # boud = [cv2.boundingRect(cnt) for cnt in contours]
    # print(len(contours))
    dt_arr=[]
    cut_cnt_arr = []
    cor_arr = []
    #dieu kien phat hien contour
    if (len(contours)>=len(contours_INV)):
        list_contours = contours
    else:
        list_contours = contours_INV
    for c in list_contours:
        place_arr = []
        return_arr = []
        x,y,w,h = cv2.boundingRect(c)
        #tinh trung vi
        dt_arr.append(w*h)
        n = len(dt_arr)
        dt_arr.sort()
        if n % 2 == 0: 
            median1 = dt_arr[n//2] 
            median2 = dt_arr[n//2 - 1] 
            median = (median1 + median2)/2
        else: 
            median = dt_arr[n//2]
    # print(median)
        if ( (0.1* median) < (cv2.contourArea(c)) < (20 * median)):
            # cv2.rectangle(img,(x,y),(x+w, y+h),(0,255,0),1)
            cor_arr.append(x)#lay toa do x
            # print(cor_arr)
            cut_cnt = img[y:y+h, x:x+w]
            cut_cnt_arr.append(cut_cnt)
            # print(cut_cnt_arr)
            # cor_arr = []
            # print(cut_cnt_arr)
        #chuyen ve dataframe de lay ra vi tri toa do
        df = pd.DataFrame(cor_arr,columns = columns)
        #sap xep toa do theo thu tu thap den lon
        df = df.sort_values('x',ascending = True)
        # print(df)
        #lay index cua toa do
        df = df.reset_index()
        for n in range (len(df)):
            place_arr.append(df.iloc[n,0])
        for place in list(place_arr):
            return_arr.append(cut_cnt_arr[place])
        # print (return_arr)
        # for l in range(len(df)):
        #     temp =[]
        #     temp.append(df.iloc[l][0])
        #     temp.append(df.iloc[l][1])
        #     temp.append(df.iloc[l][2])
        #     temp.append(df.iloc[l][3])
        #     arr_index.append(temp)
        # print(arr_index)
        # arr_index = np.array(arr_index)
    return img, return_arr

#ham reshape ảnh để hợp với model
def reshape(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    threshold = img / 255
    threshold = cv2.resize(threshold,(32,32))
    thres_arr = []
    thres_arr.append(threshold)
    thres_arr = np.array(thres_arr)
    thres_arr = thres_arr.reshape(thres_arr.shape[0], 32, 32,1)

    return thres_arr
#hàm dự báo
def predict(img):
    pre = load_model.predict(img)
    char = np.argmax(pre)
    text = class_name_text[char]
    return text


def calc(pil_image):
    result = pil_image.convert(mode='RGB', palette=Image.ADAPTIVE, colors=1)
    result.putalpha(0)
    colors = result.getcolors()

# # cut_arr = []
# img = cv2.imread("1.png")
# a = detect_east(img)
# cv2.imshow("a",a)
# cv2.waitKey(0)