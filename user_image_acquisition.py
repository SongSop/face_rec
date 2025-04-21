import cv2
import numpy as np
import os
# coding=utf-8
import urllib
import urllib.request
import hashlib

def picture_num():
    path = './data/jm/'
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    for imagePath in imagePaths:
        picturenum = int(os.path.split(imagePath)[1].split('.')[0])
    return picturenum

#摄像头
cap=cv2.VideoCapture(0)
savepath='C:/Users/song/Desktop/face/data/jm/'
falg = 1
chinese_char = input('请输入姓名：')
num = picture_num()+1
# print(chinese_char)
# unicode_str = chinese_char.encode("utf-8")
# print(unicode_str)
print("请按下s拍照")
while(cap.isOpened()):#检测是否在开启状态
    ret_flag,Vshow = cap.read()#得到每帧图像
    cv2.imshow("Capture_Test",Vshow)#显示图像
    k = cv2.waitKey(1) & 0xFF#按键判断
    if k == ord('s'):#保存
       # cv2.imwrite("C:/Users/song/Desktop/mycodetest/opencv/data/jm/" +str(num)+str(unicode_str)+".jpg", Vshow)
       cv2.imencode('.jpg', Vshow)[1].tofile(savepath+str(num)+'.'+chinese_char+'.jpg')
       print("success to save"+str(num)+".jpg")
       print("图像采集成功请按下x退出")
       num += 1
    elif k == ord('x'):#退出
        break
#释放摄像头
cap.release()
#释放内存
cv2.destroyWindow("Capture_Test")
