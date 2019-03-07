#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 12:00:12 2019

@author: vladgriguta
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 13:59:10 2019

@author: vladgriguta
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime


import os
import itertools

#ML libraries
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score


import dataPrep
import gc
gc.collect()


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix',
                          cmap=plt.cm.Blues,directory=''):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Showing normalized confusion matrix")
    else:
        print('Showing confusion matrix, without normalization')
    #print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),fontsize = 6,
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(directory+title+'.png')
    plt.gcf().clear()
    
    
def NeuralNet(trim_columns,input_table, n_jobs=-1):
    
    # Get data from database
    x, dummy_y,encoder,class_weights,_ = dataPrep.getDataNN(input_table, trim_columns)
    
    # Split data 60:20:20
    x_train, x_test, dummy_y_train, dummy_y_test = train_test_split(x, dummy_y, test_size=0.2, random_state=0)
    x_train, x_val, dummy_y_train, dummy_y_val = train_test_split(x_train, dummy_y_train, test_size=0.2, random_state=0)
    
    
    input_dim = len(x[0])
    output_dim = len(dummy_y[0])
    # Importing the Keras libraries and packages
    from keras.models import Sequential
    from keras.utils import Sequence
    from keras.layers import Dense
    from keras.layers import Dropout
    
    class DataSequenceGenerator(Sequence):
    
        def __init__(self, x_train, y_train, batch_size):
            self.x, self.y = x_train, y_train
            self.batch_size = batch_size
    
        def __len__(self):
            return int(np.ceil(len(self.x) / float(self.batch_size)))
    
        def __getitem__(self, idx):
            batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
            batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

            return np.array(batch_x), np.array(batch_y)
    
    params = {'batch_size': 128}
    training_generator = DataSequenceGenerator(x_train, dummy_y_train, **params)
    steps_train = len(x_train)/float(params['batch_size'])
    validation_generator = DataSequenceGenerator(x_val, dummy_y_val, **params)
    steps_val = len(x_val)/float(params['batch_size'])
    
    # define baseline model
    def baseline_model():
       	# create model
        model = Sequential()
        model.add(Dense(units = 5, kernel_initializer = 'uniform', activation = 'sigmoid', input_dim=input_dim))
        #model.add(Dropout(0.2))
        model.add(Dense(units=output_dim, activation='softmax'))
    	# Compile model
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        
        return model
    

    classifier = baseline_model()
    
    """
    classifier.fit_generator(generator=training_generator,steps_per_epoch=steps_train,
                             validation_data=validation_generator,validation_steps=steps_val,
                             epochs=5,use_multiprocessing=True,workers=n_jobs)
    """
    # train the model
    classifier.fit_generator(generator=training_generator,epochs=5,use_multiprocessing=True,workers=n_jobs)
    # First round of evaluation
    evaluation = classifier.evaluate(x=x_test, y=dummy_y_test, batch_size=params['batch_size'])
    print('The accuracy on test data is: a= ' + str(evaluation[1]))
    
    
    """
    # Compute the F1 Score
    f1 = f1_score(encoded_y_test,prediction_classes,average='macro')        
    # Compute and plot the confusion matrix
    #cnf_matrix = confusion_matrix(encoded_y_test, prediction_classes,labels=classes_y_test)
    acc = accuracy_score(encoded_y_test,prediction_classes)
    print('The f1 score of NN is:   '+str(f1))
    print('The accuracy of NN is:   '+str(acc))
    """
    
    # Prepare the data required for confusion matrix plot
    dummy_y_pred = classifier.predict(x_test)
    classes_y_pred,encoded_y_pred = dataPrep.decode(dummy_y_pred,encoder)
    classes_y_test,encoded_y_test = dataPrep.decode(dummy_y_test,encoder)
    
    return classes_y_test, classes_y_pred

if __name__ == "__main__":
    
    input_table = '../moreData/test_query_table_1M'
    trim_columns=['#ra', 'dec', 'z', 'peak','integr','rms','subclass','class']

    classes_y_test, classes_y_pred = NeuralNet(trim_columns,input_table,n_jobs=3)
    
    labels = ['GALAXY','QSO','STAR']
    
    cnf_matrix = confusion_matrix(classes_y_test, classes_y_pred,labels=labels)
    
    directory = 'ANN_classification_7Mar/'
    if not os.path.exists(directory):
        os.makedirs(directory)
        
    plot_confusion_matrix(cm=cnf_matrix,classes=labels,directory=directory,title='ANN_Confusion_Matrix')