#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 12:01:34 2019

@author: vladgriguta
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils

def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pd.DataFrame(pickle.load(f))


def getDataNN(input_table, trim_columns):
    """
    This function reads in the data from input table and returns the scaled
    features (x) and the encoded classes (dummy_y)
    """
    
    
    data_table=load_obj(input_table)
    
    ###########################################################################
    #############################Features######################################
    ###########################################################################
    
    #trim away unwanted columns
    x=data_table.drop(columns=trim_columns)
    name_of_features = x.columns
    # Scale all data
    scaler = MinMaxScaler()
    scaler.fit(x)
    x = scaler.transform(x)
    
    ###########################################################################
    #############################Classes#######################################
    ###########################################################################
    y=data_table['class']
    
    # encode class values as integers
    encoder = LabelEncoder()
    encoder.fit(y)
    encoded_Y = encoder.transform(y)
    # compute weights to account for class imbalace and improve f1 score
    y_cat = np.unique(encoded_Y)
    class_appearences = {y_cat[i]:np.sum(encoded_Y==y_cat[i]) for i in range(len(y_cat))}
    n_classes_norm = len(encoded_Y)/10000
    class_weights = {list(class_appearences.keys())[i]:n_classes_norm/list(class_appearences.values())[i] for i in range(len(class_appearences))}
    # convert integers to dummy variables (i.e. one hot encoded)
    dummy_y = np_utils.to_categorical(encoded_Y)
    
    return x,dummy_y,encoder,class_weights, name_of_features


def decode(dummy_y,encoder):
    """
    Function that takes the dummy variable and its encoder and transforms it
    back to the initial form
    """
    # from dummy back to class names
    encoded_y = np.zeros(len(dummy_y))
    for i in range(len(dummy_y)):
        encoded_y[i] = int(np.argmax(dummy_y[i]))
    classes_y =  encoder.inverse_transform(encoded_y.astype(int))
    
    return classes_y,encoded_y






