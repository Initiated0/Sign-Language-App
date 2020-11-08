import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import Model, Sequential, load_model


from tensorflow import keras


def get_model():
    global model
    model = Sequential()
    model = tf.keras.models.load_model('asl3.h5')
    print(' ** Model Loaded!')

print(' * Loading Keras model...')
get_model()


def preprocess_image(image):
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = tf.keras.applications.mobilenet.preprocess_input(image)

    return image

class VideoCamera(object):
    f = open("flaskblog/output.txt", "w")  #root is at run.py
    data = ""
    prev_letter = ""
    def __init__(self):
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()
    def openfile(self):
        self.f = open("flaskblog/output.txt", "w")#root is at run.py
    def closefile(self):
        self.f.close()
    def givedata(self):
        return self.data


    def get_frame(self):
        ret, img = self.video.read()
        
        cv2.rectangle(img, (250, 270), (50, 70), (255, 0, 0), 2)  # bottom right , top left + BGR
        crop_img = img[70:270, 50:250]  # first = y-axis, x-axis
        

        font = cv2.FONT_HERSHEY_SIMPLEX
        # DO WHAT YOU WANT WITH TENSORFLOW / KERAS AND OPENCV
        if ret == True:
            img2 = cv2.resize(crop_img, dsize=(64, 64))

            preprocessed_image = preprocess_image(img2)

            prediction = model.predict(preprocessed_image)
            x = np.argmax(prediction)
            labels_dict = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'J': 9, 'K': 10, 'L': 11,
                          'M': 12,'N': 13, 'O': 14, 'P': 15, 'Q': 16, 'R': 17, 'S': 18, 'T': 19, 'U': 20, 'V': 21, 'W': 22, 'X': 23,
                          'Y': 24, 'Z': 25, 'space': 26, 'del': 27, 'nothing': 28}
            key = list(labels_dict.keys())
            val = list(labels_dict.values())
            x = key[val.index(x)]

            # if x != self.prev_letter:
            #     if x == 'nothing' or x == 'space':
            #         self.f.write(' ')
            #     else:
            #         self.f.write(x)
            #     self.prev_letter = x

            if x == 'nothing':
                if self.prev_letter == 'space':
                    self.f.write(' ')
                    self.data = self.data + " "
                elif self.prev_letter != 'nothing' and self.prev_letter != 'del':
                    self.f.write(self.prev_letter)
                    self.data = self.data + self.prev_letter
                   # print("camera.py data : ", self.data)
            self.prev_letter = x

            cv2.putText(img, x, (100, 100), font, 1, (0, 0, 255), 2)
            #cv2.rectangle(img, (300,300),(50,50),(255,0,0),2)

            ret, jpeg = cv2.imencode('.jpg', img)
            return jpeg.tobytes()
        else:
            #self.f.close()
            return None


