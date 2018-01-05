# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

train_data= pd.read_csv("../input/train.csv")
test_data= pd.read_csv("../input/test.csv")
print(np.shape(train_data))


X_train, y_train= train_data.drop(["label"], axis=1).values, train_data["label"]
X_test= test_data.values


print(np.shape(X_test))
print(np.shape(X_train))
print(np.shape(y_train))


import keras
from keras.utils import to_categorical

y_train= to_categorical(y_train, num_classes=10)

print(np.shape(y_train))

from keras.layers import Input, Dense, Activation
from keras.models import Sequential
import keras.layers as ll

num_classes= 10
model= Sequential(name="hackdtv")
model.add(ll.InputLayer([784]))
mean= X_train.mean(axis=0)
std= X_train.std(axis=0) + 1e-5
model.add(ll.Lambda(lambda pix, mu, std: (pix - mu) / std,
                    arguments={'mu': mean, 'std': std}))
#model.add(ll.Flatten())
from keras import regularizers
model.add(ll.Dense(512, input_dim=784,  kernel_regularizer=regularizers.l2(0.001)))
model.add(ll.Activation('relu'))

model.add(ll.Dense(512, input_dim=512, kernel_regularizer=regularizers.l2(0.001)))
model.add(ll.Activation('relu'))

model.add(ll.Dense(num_classes))
model.add(ll.Activation('softmax'))
model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, batch_size=256, epochs=20, verbose=1, validation_split=0.1)
score= model.predict(X_test)
predictions= np.argmax(score, axis=1)
pred_data= {"ImageId": np.arange(1, len(predictions)+1), "Label": predictions}
labels_csv= pd.DataFrame(pred_data).to_csv("submit.csv", columns = ["ImageId", "Label"], index=False)

