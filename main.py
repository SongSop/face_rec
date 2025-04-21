import cv2
import numpy as np
import os
# coding=utf-8
import urllib
import urllib.request
import hashlib
from PIL import Image, ImageDraw, ImageFont

#加载训练数据集文件
recogizer=cv2.face.LBPHFaceRecognizer_create()
recogizer.read('trainer/trainer.yml')
names=[]
warningtime = 0
printnum = 0
identifynum = 0
#准备识别的图片
def face_detect_demo(img):
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)#转换为灰度
    face_detector=cv2.CascadeClassifier('C:/Users/song/Downloads/opencv/sources/data/haarcascades/haarcascade_frontalface_alt.xml')
    face=face_detector.detectMultiScale(gray,1.1,5,cv2.CASCADE_SCALE_IMAGE,(100,100),(300,300))
    #face=face_detector.detectMultiScale(gray)
    global printnum
    for x,y,w,h in face:
        cv2.rectangle(img,(x,y),(x+w,y+h),color=(0,0,255),thickness=2)
        # 人脸识别
        ids, confidence = recogizer.predict(gray[y:y + h, x:x + w])
        #print('标签id:',ids,'置信评分：', confidence)
        if confidence > 60:
            global warningtime

            warningtime += 1
            if warningtime > 100 and printnum==0:
                print("请按下k进行用户图像采集")
                printnum=printnum+1
            if (warningtime > 100) and (ord('k') == cv2.waitKey(10)):
               cap.release()
               cv2.destroyWindow("result")
               os.system('python user_image_acquisition.py')
               os.system('python training_data_update.py')
               printnum=0
               warningtime = 0
               cv2.imshow('result', img)
            cv2.putText(img, 'unkonw', (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 1)
        else:
            global identifynum
            identifynum += 1
            # cv2.putText(img,str(names[ids-1]), (int(x + 10),int(y - 10) ), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 1)
            img = cv2AddChineseText(img,str(names[ids-1]),(x, y), (0, 255, 0),30 )
            if identifynum > 100 and printnum==0:
                print("检测到当前用户已存在就诊记录请按下a显示用户姓名")
                printnum=printnum+1
            if (identifynum > 100) and (ord('a') == cv2.waitKey(10)):
                print(str(names[ids-1]))
            # print(x,y)
    cv2.imshow('result',img)
    #print('bug:',ids)

def name():
    path = './data/jm/'
    #names = []
    imagePaths=[os.path.join(path,f) for f in os.listdir(path)]
    for imagePath in imagePaths:
       name = str(os.path.split(imagePath)[1].split('.',2)[1])
       names.append(name)


def cv2AddChineseText(img, text, position, textColor=(0, 255, 0), textSize=30):
    if (isinstance(img, np.ndarray)):  # 判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # 创建一个可以在给定图像上绘图的对象
    draw = ImageDraw.Draw(img)
    # 字体的格式
    fontStyle = ImageFont.truetype(
        "simsun.ttc", textSize, encoding="utf-8")
    # 绘制文本
    draw.text(position, text, textColor, font=fontStyle)
    # 转换回OpenCV格式
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

# cap=cv2.VideoCapture(0)
cap=cv2.VideoCapture(0)
name()
while True:
    flag,frame=cap.read()
    if not flag:
        break
    face_detect_demo(frame)
    if ord(' ') == cv2.waitKey(10):
        break
cv2.destroyAllWindows()
cap.release()
# print(names)


