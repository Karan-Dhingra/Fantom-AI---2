import cv2
import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


class Video(object):
    
    def __init__(self):
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        # releasing video
        self.video.release()

    def get_frame(self):
        emotion_model = Sequential()
        emotion_model.add(Conv2D(32, kernel_size=(
            3, 3), activation='relu', input_shape=(48, 48, 1)))
        emotion_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
        emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
        emotion_model.add(Dropout(0.25))
        emotion_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
        emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
        emotion_model.add(Dropout(0.25))
        emotion_model.add(Flatten())
        emotion_model.add(Dense(1024, activation='relu'))
        emotion_model.add(Dropout(0.5))
        emotion_model.add(Dense(7, activation='softmax'))
        # emotion_model.load_weights('F:\\Projects\\LATESTT\\ai-camera\\src\\model.h5')
        emotion_model.load_weights('modelnew.h5')
        # emotion_model.load_weights('modelnew.h5')
        cv2.ocl.setUseOpenCL(False)

        emotion_dict = {
            0: "Angry",
            1: "Disgusted",
            2: "Fearful",
            3: "Happy",
            4: "Sad",
            5: "Neutral",
            6: "Surprise",
        }

        show_text = [0]

        ret, img = self.video.read()
        img = cv2.resize(img, (600, 500))
        imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = faceDetect.detectMultiScale(img, 1.3, 5)

        for x, y, w, h in faces:
            x1, y1 = x+w, y + h
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 255), 1)
            cv2.line(img, (x, y), (x+30, y), (255, 0, 255), 6)  # Top Left
            # cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 255), 1)
            cv2.line(img, (x, y), (x, y+30), (255, 0, 255), 6)

            cv2.line(img, (x1, y), (x1-30, y), (255, 0, 255), 6)  # Top Right
            cv2.line(img, (x1, y), (x1, y+30), (255, 0, 255), 6)

            cv2.line(img, (x, y1), (x+30, y1), (255, 0, 255), 6)  # Bottom Left
            cv2.line(img, (x, y1), (x, y1-30), (255, 0, 255), 6)

            cv2.line(img, (x1, y1), (x1-30, y1),
                     (255, 0, 255), 6)  # Bottom right
            cv2.line(img, (x1, y1), (x1, y1-30), (255, 0, 255), 6)

            # cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            roi_gray_frame = imgGray[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(
                cv2.resize(roi_gray_frame, (48, 48)), -1), 0)
            font = cv2.FONT_HERSHEY_SIMPLEX
            prediction = emotion_model.predict(cropped_img)

            print('PREDICTION', prediction)
            maxindex = int(np.argmax(prediction))
            print('MAX-INDEX', maxindex)

            print(emotion_dict[maxindex])
            cv2.putText(img, emotion_dict[maxindex], (x+20, y-60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            show_text[0] = maxindex

        ret, jpg = cv2.imencode('.jpg', img)
        return jpg.tobytes()
