from json import load
import os
import cv2
import tensorflow as tf
import os
import numpy as np

from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from tensorflow.python import metrics

model = load_model("./static/temporary.h5")
def testing():
        
    webb_cam = cv2.VideoCapture(0)

    NumberOfFrames = 0
    
   # model = load_model("./static/temporary.h5")
    while True:
        captured = False

        ret , frame = webb_cam.read()

        #cv2.imshow("webcam" , frame)

        cv2.imwrite("./static/testing_images/frame" + str(NumberOfFrames) + ".jpg" , frame)

        NumberOfFrames += 1

        print(NumberOfFrames)

        if(NumberOfFrames == 1):
            captured = True
            break
            
    webb_cam.release()
    cv2.destroyAllWindows()
        
    if captured == True:
        # Testing image directory
        testing_image_directory = './static/testing_images'

        # All image files in the directory
        img_files = os.listdir(testing_image_directory)

       

        ## ADD CODE HERE
        
        for file in img_files:
            img_files_path = os.path.join(testing_image_directory,file)
            #load image
            img_1 = load_img(img_files_path , target_size = (180 , 180))
            #convert img into array 
            img_2 = img_to_array(img_1)
            img_3 = np.expand_dims(img_2 ,axis = 0)
            prediction = model.predict(img_3)
            #print(prediction)
            print(prediction)
            predicted_out = "this is fake"
            predict_class = np.argmax(prediction , axis = 1)
            if predict_class[0] == 0:
                 predicted_out = "class1"
            elif predict_class[0] == 1:
                 predicted_out = "class2"
            #plot the img
            
    return predicted_out