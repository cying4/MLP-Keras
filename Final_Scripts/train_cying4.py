import os
import numpy as np
import cv2
from sklearn.preprocessing import LabelEncoder
#%%
if "train" not in os.listdir():
    os.system("wget https://storage.googleapis.com/exam-deep-learning/train.zip")
    os.system("unzip train.zip")

DATA_DIR = os.getcwd() + "/train/"
RESIZE_TO = 100
#%%
x, y = [], []
for path in [f for f in os.listdir(DATA_DIR) if f[-4:] == ".png"]:
    x.append(cv2.resize(cv2.imread(DATA_DIR + path), (RESIZE_TO, RESIZE_TO)))
    with open(DATA_DIR + path[:-4] + ".txt", "r") as s:
        label = s.read()
    y.append(label)
x, y = np.array(x), np.array(y)
le = LabelEncoder()
le.fit(["red blood cell", "ring", "schizont", "trophozoite"])
y = le.transform(y)
print(x.shape, y.shape)
#%%
from keras.utils import to_categorical
y = to_categorical(y, num_classes=4)

#%%
from sklearn.preprocessing import scale
x=x.reshape(len(x),-1)
x=scale(x)
#%%
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42, test_size=0.2, stratify=y)
#%%
from keras.initializers import glorot_uniform
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, BatchNormalization, Activation
from keras.optimizers import Adam,Nadam
from keras.callbacks import EarlyStopping
from keras.optimizers import SGD
from keras.optimizers import RMSprop
model = Sequential()
model.add(Dense(2000, input_dim=30000, activation='relu'))
model.add(Dropout(0.25))
#model.add(BatchNormalization())
model.add(Dense(2000, activation='relu'))
model.add(Dropout(0.25))
#model.add(BatchNormalization())
model.add(Dense(4000, activation='relu'))
model.add(Dropout(0.25))
#model.add(BatchNormalization())
model.add(Dense(5000, activation='relu'))
model.add(Dropout(0.25))
#model.add(BatchNormalization())
#model.add(Dense(500, activation='relu'))
#model.add(Dropout(0.2))
#model.add(BatchNormalization())
model.add(Dense(4, activation = 'sigmoid'))
model.compile(loss='BinaryCrossentropy', optimizer=SGD(lr=0.01), metrics=['Accuracy'])
#%%
from keras.callbacks import ModelCheckpoint
es = EarlyStopping(monitor='val_loss',mode='min',patience=5,restore_best_weights=True)
mc = ModelCheckpoint('mlp_cying4.h5', monitor='val_loss', mode='max',save_best_only=True)
model.fit(x_train, y_train, batch_size=512, epochs=2000, validation_data=(x_test, y_test), callbacks=[es, mc])
#%%
from sklearn.metrics import cohen_kappa_score, f1_score
print("Final accuracy on validations set:", 100*model.evaluate(x_test, y_test)[1], "%")
print("Cohen Kappa", cohen_kappa_score(np.argmax(model.predict(x_test),axis=1),np.argmax(y_test,axis=1)))
print("F1 score", f1_score(np.argmax(model.predict(x_test),axis=1),np.argmax(y_test,axis=1), average = 'macro'))
#%%
model.save('mlp_cying4.hdf5')