import os
from tensorflow.keras.preprocessing import image
from PIL import Image
import numpy as np
import tensorflow as tf
# import cv2

categories = os.listdir("Data-Sets/dataset_for_model/train")
categories.sort()
print(categories)

# load the saved model
modelSavedPath = "Data-Sets/dataset_for_model/MoonV3.h5"
model = tf.keras.models.load_model(modelSavedPath)
model.summary()

# predict the image
def classify_image(imageFile):
    x = []

    img = Image.open(imageFile)
    img.load()
    img = img.resize((320,320), Image.ANTIALIAS)

    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    print(x.shape)
    pred = model.predict(x)
    print(pred)

img = "test.jpg"
classify_image(img)