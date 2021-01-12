import glob
import os
from collections import Counter

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             plot_confusion_matrix)
from sklearn.model_selection import GridSearchCV
from tensorflow import keras
from sklearn.model_selection import train_test_split


seed=0
np.random.seed(seed)
tf.random.set_seed(seed)

num_classes = 62 # 62 lop([0-9],[A-Z],[a-z])
img_rows, img_cols = 32,32 # kich thuoc anh


#khai bao nhan
class_name_dig = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 
'10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27',
'28', '29', '30', '31', '32', '33', '34', '35', 
'36', '37', '38', '39', '40', '41', '42', '43', '44', '45', '46', '47', '48', '49', '50', '51', '52', '53', 
'54', '55', '56', '57', '58', '59', '60', '61']

class_name_text = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 
'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R',
'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 
'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 
's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
#doc tat ca hinh anh trong thu muc
samples = []
sizes = []
spl = "Sample0"
for i in range(1,63):
    # print(i)
    if (i < 10):
        str_tmp = "0" + str(i)
        spl_app = spl + str_tmp
        samples.append(spl_app)
    else:
        spl_app = spl + str(i)
        samples.append(spl_app)


#ham gan nhan
def get_text_data(path):
    img_list = [] #mang chua hinh
    label_list = [] #mang chua nhan
    for i in range(len(samples)):
        for img_org_path in glob.iglob(path + str(samples[i]) + '/*.png'):
            img = cv2.imread(img_org_path, 0)#doc va chuyen anh ve xam
            img = cv2.resize(img,(32,32))#resize anh(143.143)
            # img = np.full((100,80,3), 12, np.uint8)
            img = np.array(img, dtype=np.uint8)
            img_list.append(img)
            label_list.append(np.uint8(class_name_dig[i]))
    return img_list, label_list


    
root_path = './Imgdataset/GoodImg/Bmp/' #duong dan den thu muc hinh anh train
dig, lab = get_text_data(root_path)

# root_path_test = './Img/GoodImg/Bmp_test/' #duong dan den thu muc test
# dig_test, lab_test = get_text_data(root_path_test)
#phan chia tap du lieu
X_train, X_test, Y_train, Y_test = train_test_split(dig, lab, test_size = 0.2, random_state = 100, shuffle = True)



#chuyen ve mang
#tap train
X_train = np.array(X_train)
Y_train = np.array(Y_train)
#tap test
X_test = np.array(X_test)
Y_test = np.array(Y_test)
#thay doi hinh dang
X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols,1) 
X_train = X_train.astype('float32')
X_train /= 255

X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols,1)
X_test = X_test.astype('float32')
X_test /= 255

#one hot encoding
Y_train = keras.utils.to_categorical(Y_train, num_classes)
Y_test = keras.utils.to_categorical(Y_test, num_classes)


from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from keras.models import Sequential

#khoi tao model
model = Sequential()
model.add(Conv2D(32,(3,3), input_shape = (32,32,1), activation='relu'))
model.add(Conv2D(32,(3,3),activation='relu')),
model.add(MaxPooling2D(pool_size=(2,2))),
model.add(Dropout(0.25)),
model.add(Conv2D(64,(3,3),activation='relu')),
model.add(Conv2D(64,(3,3),activation='relu')),
model.add(MaxPooling2D(pool_size=(2,2))),
model.add(Dropout(0.25)),
model.add(Flatten(input_shape=(32,32))), #chuyển về vector
model.add(Dense(128, activation='relu')),
model.add(Dropout(0.5)),
model.add(Dense(62))

#train model
model.compile(optimizer='adam', loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True), metrics=['accuracy'])#train the model
# model.summary()
model.fit(X_train, Y_train, epochs=50)# evaluate the model

#luu model
# filename = 'modelkeras_conv.h5'
# keras.models.save_model(model, filename)