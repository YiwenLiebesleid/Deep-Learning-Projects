from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)
simplefilter(action='ignore', category=Warning)

import numpy as np
import tensorflow as tf
import cv2
from keras.layers.convolutional import ZeroPadding2D, AveragePooling2D
from keras.models import Model
from keras.layers import Dense, Input, Flatten, Dropout
from keras.applications.mobilenet import MobileNet
from keras.optimizers import Adam
import face_recognition
from PIL import Image


# load model

input_tensor = Input(shape=(224, 224, 3))
model = MobileNet(input_tensor = input_tensor, include_top=False, input_shape=(224, 224, 3), weights='imagenet')
x = model.output
x = ZeroPadding2D()(x)
x = AveragePooling2D((3,3))(x)
x = Flatten()(x)
x = Dropout(0.3)(x)
predictions = Dense(10, activation=None)(x)
model = Model(inputs=input_tensor, outputs=predictions)

model.compile(optimizer=Adam(lr=0.0005),loss='mse',metrics=[tf.keras.metrics.MeanAbsoluteError(),tf.keras.metrics.RootMeanSquaredError()])

model.load_weights('faceDetect-5-4.h5')


# capture camera

cap = cv2.VideoCapture(0)
while(True):
    ret, frame = cap.read()
    faces = face_recognition.face_locations(frame)
    fraw = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = frame
    for (top,right,bottom,left) in faces:
        img = cv2.rectangle(img, (left,top),(right,bottom),(0,255,0),2)

        f1 = fraw[top:bottom, left:right]
        rawh, raww = f1.shape[0:2]
        ratioh, ratiow = 224.0 / rawh, 224.0 / raww
        f1 = Image.fromarray(f1)
        f1 = f1.resize((224, 224))
        f1 = np.array([np.array(f1)])
        pre = model.predict(f1 / 255.0)
        f1 = f1[0]
        pre = pre[0]
        pre[0::2] /= ratiow
        pre[1::2] /= ratioh
        pre = pre.astype(np.int)
        pre[0::2] += left
        pre[1::2] += top

        for i in range(len(pre) // 2):
            p1, p2 = pre[i*2], pre[i*2+1]
            cv2.circle(img, (p1,p2), 1, (0,0,255), 2)
        break

    cv2.imshow('frame2',img)
    if cv2.waitKey(5) & 0xFF == ord('q'):       # press 'q' to exit
        break

cap.release()
cv2.destroyAllWindows()
