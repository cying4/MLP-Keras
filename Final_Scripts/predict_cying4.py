import os
import numpy as np
from keras.models import load_model
import random
import cv2
from sklearn.preprocessing import scale
from keras.optimizers import RMSprop
RESIZE_TO = 100
def predict(x):
    images=[]
    for i in x:
        images.append(cv2.resize(cv2.imread(i), (RESIZE_TO, RESIZE_TO)))
    test=np.array(images)
    test = test.reshape(len(test), -1)
    test = scale(test)
    model = load_model('mlp_cying4.hdf5')
    y_pred = np.argmax(model.predict(test), axis=1)
    return y_pred, model
