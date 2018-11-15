
# coding: utf-8

# In[35]:


from time import ctime
from time import time
import matplotlib.pyplot as plt
import math
import pandas as pd
import numpy as np
import random
import io
import keras
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn import metrics  
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import StratifiedKFold
from sklearn.cross_validation import cross_val_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.utils import np_utils  
from keras.optimizers import RMSprop 
from keras.wrappers.scikit_learn import KerasClassifier
from keras.layers.normalization import BatchNormalization
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt


#Read the training file and record the running time
time_1 = time()
raw_data = pd.read_csv('covtype.txt', header=None).values
time_2 = time()
print('read data cost '+ str(time_2 - time_1)+' second')

training_data = raw_data[:15120]
training_data_1 = training_data[:,:-1]
training_data_label = training_data[:,-1]
testing_data = raw_data[15120:]
testing_data_1 = testing_data[:,:-1]
testing_data_label = testing_data[:,-1]

#Normalize training and testing Data
scaler = MinMaxScaler(feature_range=(0, 1))
rescaledX = scaler.fit_transform(training_data_1)
rescaledX_test = scaler.fit_transform(testing_data_1)
np.set_printoptions(precision=3)

#Normalize the label data
for i in range(len(training_data_label)):
    training_data_label[i] = training_data_label[i] - 1
for i in range(len(testing_data_label)):
    testing_data_label[i] = testing_data_label[i] - 1

#LabelEncoder normalize the labels
encoder = LabelEncoder()
encoded_label = encoder.fit_transform(training_data_label)
training_data_label = np_utils.to_categorical(encoded_label) 

#PCA to decomposition 
pca = PCA(n_components=0.95,whiten=True)
train_x  = pca.fit_transform(rescaledX)
test_x = pca.transform(rescaledX_test)

#Define the model
def create_model():
    model_F = Sequential()
    model_F.add(Dense(units=200, input_dim=len(train_x[0]), kernel_initializer='normal'))
    model_F.add(BatchNormalization())
    #model_F.add(Dropout(0.5))
    model_F.add(Activation('relu'))
    model_F.add(Dense(units=200,kernel_initializer='normal'))
    model_F.add(BatchNormalization())
    #model_F.add(Dropout(0.5))
    model_F.add(Activation('relu'))
    model_F.add(Dense(units=7, kernel_initializer='normal', activation='softmax'))
    print(model_F.summary())
    #set the optimizer and relevant parameters, the loss funcion is categorical_crossentropy
    Nadam = keras.optimizers.Nadam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
    model_F.compile(loss='categorical_crossentropy', optimizer=Nadam, metrics=['accuracy'])
    return model_F

model_2 = KerasClassifier(build_fn=create_model, batch_size=60,epochs=100,shuffle=True,validation_split=0.1)
history_2 = model_2.fit(train_x,training_data_label)

# summarize history for accuracy
plt.plot(history_2.history['acc'])
plt.plot(history_2.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['acc', 'val_acc'], loc='upper left')
plt.show()

#predict label of each test data
predicted = model_2.predict(test_x)
predicted_prob = model_2.predict_proba(test_x)

# get accuracy value
accuracy = accuracy_score(testing_data_label, predicted)
print(accuracy)
# get precision,recall,fscore,support value
precision, recall, fscore, support = precision_recall_fscore_support(testing_data_label, predicted)

# draw decent confusion matrix
print(confusion_matrix(testing_data_label, predicted))
print(classification_report(testing_data_label, predicted))

print(pd.crosstab(testing_data_label, predicted, rownames = ['label'], colnames = ['predict']))

y_one_hot = label_binarize(testing_data_label, classes=[0,1,2,3,4,5,6])
y_score = predicted_prob
metrics.roc_auc_score(y_one_hot, y_score)  
fpr, tpr, thresholds = metrics.roc_curve(y_one_hot.ravel(),y_score.ravel()) 
auc=metrics.auc(fpr, tpr) 
plt.plot(fpr, tpr,label='average ROC curve of fold %d (area = %0.2f)' % (i, auc), color='r', linestyle='-', linewidth=2)

plt.xlim((-0.01, 1.10))  
plt.ylim((-0.01, 1.10))  
plt.xticks(np.arange(0, 1.1, 0.1))  
plt.yticks(np.arange(0, 1.1, 0.1))  
plt.xlabel('False Positive Rate', fontsize=13)  
plt.ylabel('True Positive Rate', fontsize=13)  
plt.grid(b=True, ls=':')  
plt.legend(loc='lower right', fancybox=True, framealpha=0.8, fontsize=9)  
plt.title('ROC and AUC of NN', fontsize=17)  
plt.show()  

