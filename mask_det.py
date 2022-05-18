

#import multiprocessing
import numpy as np
import cv2
import tensorflow.keras as tf
import pyttsx3
import math
import os

DIR_PATH = os.path.dirname(os.path.realpath(__file__))



def main():

    labels_path = f"{DIR_PATH}/model/labels.txt"
    labelsfile = open(labels_path, 'r')

    classes = []
    line = labelsfile.readline()
    while line:
        classes.append(line.split(' ', 1)[1].rstrip())
        line = labelsfile.readline()
    labelsfile.close()

    model_path = f"{DIR_PATH}/model/keras_model.h5"
    model = tf.models.load_model(model_path, compile=False)

    cap = cv2.VideoCapture(0)

    frameWidth = 1280
    frameHeight = 720

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frameWidth)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frameHeight)
   
    cap.set(cv2.CAP_PROP_GAIN, 0)

    while True:

       
        np.set_printoptions(suppress=True)

  
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

        # capture image
        check, frame = cap.read()
        # mirror image - mirrored by default in Teachable Machine
      
        frame = cv2.flip(frame, 1)

        # crop to square for use with TM model
        margin = int(((frameWidth-frameHeight)/2))
        square_frame = frame[0:frameHeight, margin:margin + frameHeight]
        # resize to 224x224 for use with TM model
        resized_img = cv2.resize(square_frame, (224, 224))
        # convert image color to go to model
        model_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)

        # turn the image into a numpy array
        image_array = np.asarray(model_img)
        # normalize the image
        normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
        # load the image into the array
        data[0] = normalized_image_array

        # run the prediction
        predictions = model.predict(data)

        # confidence threshold is 90%.
        conf_threshold = 90
        confidence = []
        conf_label = ""
        threshold_class = ""
       
        per_line = 2  # number of classes per line of text
        bordered_frame = cv2.copyMakeBorder(
            square_frame,
            top=0,
            bottom=30 + 15*math.ceil(len(classes)/per_line),
            left=0,
            right=0,
            borderType=cv2.BORDER_CONSTANT,
            value=[0, 0, 0]
        )
        # for each one of the classes
        for i in range(0, len(classes)):
            confidence.append(int(predictions[0][i]*100))
            if (i != 0 and not i % per_line):
                cv2.putText(
                    img=bordered_frame,
                    text=conf_label,
                    org=(int(0), int(frameHeight+25+15*math.ceil(i/per_line))),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.5,
                    color=(255, 255, 255)
                )
                conf_label = ""
            # append classes and confidences to text for label
            conf_label += classes[i] + ": " + str(confidence[i]) + "%; "
            # prints last line
            if (i == (len(classes)-1)):
                cv2.putText(
                    img=bordered_frame,
                    text=conf_label,
                    org=(int(0), int(frameHeight+25+15*math.ceil((i+1)/per_line))),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.5,
                    color=(255, 255, 255)
                )
                conf_label = ""
            if confidence[i] > conf_threshold:
                threshold_class = classes[i]
        # add label class above confidence threshold
        cv2.putText(
            img=bordered_frame,
            text=threshold_class,
            org=(int(0), int(frameHeight+20)),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.75,
            color=(255, 255, 255)
        )

        # original video feed implementation
        cv2.imshow("Capturing", bordered_frame)
        cv2.waitKey(10)

       
        



if __name__ == '__main__':
    main()
