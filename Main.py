from numpy.core.arrayprint import FloatingFormat
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import img_to_array
from tensorflow.keras.utils import array_to_img
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from keras_preprocessing import image
from keras.preprocessing.image import load_img
from PIL import Image
import numpy
import statistics as stats
from datetime import datetime
import time

problem_file = "Train.csv"
X_traintmp = []
y_traintmp = []
X_testtmp = []
y_testtmp = []
imgsize = 100
imgtraincount = 3406
imgvalcount = 3406
try:
    with open(problem_file, 'r') as f:
        f.readline()
        tmpcounter = 0
        for line in f:
            Image_ID = ""
            clss = ""
            xmin = 0.0
            ymin = 0.0
            width = 0.0
            height = 0.0

            tmpstring = ""
            count = 0
            for c in line:
                if c != ",":
                    tmpstring += c
                else:
                    if count == 0:
                        Image_ID = tmpstring
                        count += 1
                        tmpstring = ""
                    elif count == 1:
                        clss = tmpstring
                        count += 1
                        tmpstring = ""
                    elif count == 2:
                        xmin = float(tmpstring)
                        count += 1
                        tmpstring = ""
                    elif count == 3:
                        ymin = float(tmpstring)
                        count += 1
                        tmpstring = ""
                    elif count == 4:
                        width = float(tmpstring)
                        count += 1
                        tmpstring = ""
                    elif count == 5:
                        height = float(tmpstring)
                        count += 1
                        tmpstring = ""
            height = float(tmpstring)
            imgpath = "Train_Images/" + Image_ID + ".jpg"
            img = load_img(imgpath)
            img1 = img.crop((xmin, ymin, width+xmin, height+ymin))
            img2 = img1.resize((imgsize,imgsize), Image.ANTIALIAS)
            imgarr = tf.keras.preprocessing.image.img_to_array(img2)
            if tmpcounter < imgtraincount:
                X_traintmp.append(imgarr)
            else:
                X_testtmp.append(imgarr)
            tmpclss = 0
            if clss == "fruit_healthy":
                tmpclss = 0
            elif clss == "fruit_brownspot":
                tmpclss = 1
            else:
                tmpclss = 2
            if tmpcounter < imgtraincount:
                y_traintmp.append(tmpclss)
            else:
                y_testtmp.append(tmpclss)
            tmpcounter += 1
except IOError as ioe:
    print("The file " + problem_file + " cannot be found. " + "Please check the details provided.", ioe)

#reshape data to fit model
X_train = numpy.array(X_traintmp)
# X_val = numpy.array(X_valtmp)
X_test = numpy.array(X_testtmp)

#one-hot encode target column
y_train = numpy.array(y_traintmp)
y_test = numpy.array(y_testtmp)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

results = open("results.txt", "a")
results.write("Train-----TrainError-----Test-----Time\n")

#create model
start = time.time()
model = Sequential()
#add model layers
model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(imgsize,imgsize,3)))
model.add(Conv2D(32, kernel_size=3, activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(32, kernel_size=3, activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(3, activation='softmax'))

#compile model using accuracy to measure model performance
model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.001,beta_1=0.9,beta_2=0.999,epsilon=1e-07,amsgrad=False,name='Adam'), loss='categorical_crossentropy', metrics=['accuracy'])

# #train the model
history = model.fit(X_train, y_train, epochs=60)
end = time.time()
elapsedTime = end - start
elapsedTime = elapsedTime/60

out = model.predict(X_test[:1000])
outstr = []
tarstr = []

for x in out:
    outstr.append(str(round(x[0])) + str(round(x[1])) + str(round(x[2])))
for x in y_test:
    tarstr.append(str(round(x[0])) + str(round(x[1])) + str(round(x[2])))

correct = 0
for i in range(len(outstr)):
    if outstr[i] == tarstr[i]:
        correct += 1

accuracy2 = correct/len(out)*100
results.write(str(history.history['accuracy'][-1]*100) + "-----" + str(history.history['loss'][-1]) + "-----" + str(accuracy2) + "-----" + str(elapsedTime)+"\n")
results.close() 