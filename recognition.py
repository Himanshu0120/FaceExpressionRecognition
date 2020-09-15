from keras.models import load_model
import cv2
import numpy as np
#(0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral).
model= load_model('model.h5')

cap=cv2.VideoCapture(0)
facecascade = cv2.CascadeClassifier('faceCascade.xml')
while True:
    _,img=cap.read()
    gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=facecascade.detectMultiScale(gray,1.1,3)
    for x,y,w,h in faces:
        cv2.rectangle(img,(x-10,y-10),(x+w+10,y+h+10),(0,0,0),2)
        face=img[y-10:y+h+10,x-10:x+w+10]
        cv2.imshow('face',face)
    
    input=cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)
    input=cv2.resize(input,(48,48))
    input=input.reshape((1,48,48,1))/255
    result=np.argmax(model.predict(input))
    map={0:'Angry',1:'Disgust',2:'Fear',3:'Happy',4:'Sad',5:'Surprise',6:'Normal'}
    cv2.putText(img,map[result],(20,50),cv2.FONT_HERSHEY_TRIPLEX,3,color=(50,50,50))
    cv2.imshow('img',img)
    if cv2.waitKey(1) & 0xFF==ord('a'):
        break