from flask import Flask
from flask import Flask, request, render_template,redirect, url_for,session
import os
from os.path import join, dirname, realpath
import requests
from werkzeug.utils import secure_filename
from east_detect import *
import cv2
import keras
import re
import base64
import scipy.misc
from PIL import Image
import uuid
import json
import numpy as np


UPLOAD_FOLDER = './static/uploads/' #duong dan luu hinh anh

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg']) #tep tin duoc cho phep


app = Flask(__name__)
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

#trang chu
@app.route('/')
def home_page():
    return render_template('index.html')

#ham cho phep nhap vào là file
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

#trang xu ly
@app.route('/uploads', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # kiem tra file
        if 'file' not in request.files:
            errors = 'Vui lòng chọn tệp tin hình ảnh!'
            return render_template('index.html', errors=errors)
        file = request.files['file']
        # kiem tra file rong
        if file.filename == '':
            errors = 'Vui lòng chọn tệp tin hình ảnh!'
            return render_template('index.html', errors=errors)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = (os.path.join(app.config['UPLOAD_FOLDER'], "av"+filename))
            file.save(file_path)
            img = cv2.imread(file_path)
            # print(img)
            #detect text tu hinh anh
            east,cut_arr = detect_east(img)
            im = Image.fromarray(east)
            #save vao folder
            uuid.uuid4()
            filename_east = (str(uuid.uuid4())+".png")
            file_path_east = (os.path.join(app.config['UPLOAD_FOLDER'], filename_east))
            im.save(file_path_east)
            #cut cac phan detect
            filepath_cut_arr = []# mang hua duong dan den cac hinh sau khi tach
            filepath_cut_cnt_arr = []#mang chua cac ky tu
            text_arr = []#mang luu text
            #lay ra tat ca nhung hinh sau khi detect
            for i in range(len(cut_arr)):
                cut_im = cv2.cvtColor(cut_arr[i],cv2.COLOR_BGR2RGB)
                cut_im = Image.fromarray(cut_im)
                # print(type(east))
                #save vao folder
                filename_cut = (str(uuid.uuid4())+".png")
                file_path_cut = (os.path.join(app.config['UPLOAD_FOLDER'], filename_cut))
                cut_im.save(file_path_cut)
                filepath_cut_arr.append(file_path_cut)
                #xu ly anh
                img_pre = cv2.imread(file_path_cut)
                proc,cut_cnt_arr = pre_pro(img_pre)
                img_pro = Image.fromarray(proc)
                filename_proc = (str(uuid.uuid4())+".png")
                file_path_proc = (os.path.join(app.config['UPLOAD_FOLDER'], filename_proc))
                # filepath_pro_arr.append(file_path_proc)
                img_pro.save(file_path_proc)
                #lay ra tung ky tu
                for n in range(len(cut_cnt_arr)):
                    cut_cnt = Image.fromarray(cut_cnt_arr[n])
                    # print(cut_cnt_arr[n])
                    filename_cut_cnt = (str(uuid.uuid4())+".png")
                    file_path_cut_cnt = (os.path.join(app.config['UPLOAD_FOLDER'], filename_cut_cnt))
                    filepath_cut_cnt_arr.append(file_path_cut_cnt)
                    cut_cnt.save(file_path_cut_cnt)
                    #reshape ảnh để phù hợp với model
                    img_resh = cv2.imread(file_path_cut_cnt)
                    img_reshape = reshape(img_resh)
                    #dự đoán
                    text = predict(img_reshape)
                    text_arr.append(text)
                num_cut_cnt = len(filepath_cut_cnt_arr)
                num_text = len(text_arr)
            num_cut = len(filepath_cut_arr)
            oke_text = np.array(text_arr)
    return render_template('uploads.html', img = img, file_path = file_path, file_path_east = file_path_east,file_path_cut=file_path_cut,filepath_cut_arr=filepath_cut_arr,num_cut=num_cut,file_path_proc=file_path_proc,filepath_cut_cnt_arr=filepath_cut_cnt_arr,num_cut_cnt=num_cut_cnt,text_arr=text_arr,num_text=num_text,oke_text=oke_text)
if __name__ == '__main__':
    app.run(debug=True)