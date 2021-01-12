from imutils.object_detection import non_max_suppression
import numpy as np
import cv2
import pandas as pd
from PIL import Image
east = "frozen_east_text_detection.pb"
#ham detect van ban
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
