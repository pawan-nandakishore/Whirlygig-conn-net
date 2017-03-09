#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 19:03:54 2017

@author: pavan
"""

"""
A weighted version of categorical_crossentropy for keras (1.1.0). This lets you apply a weight to unbalanced classes.
@url: https://gist.github.com/wassname/ce364fddfc8a025bfab4348cf5de852d
@author: wassname
"""
from keras import backend as K
class weighted_categorical_crossentropy(object):
    """
    A weighted version of keras.objectives.categorical_crossentropy
    
    Variables:
        weights: numpy array of shape (C,) where C is the number of classes
    
    Usage:
        loss = weighted_categorical_crossentropy(weights).loss
        model.compile(loss=loss,optimizer='adam')
    """
    
    def __init__(self,weights):
        self.weights = K.variable(weights)
        
    def loss(y_true, y_pred):
        weights = np.ones(5)
        # scale preds so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        # clip
        y_pred = K.clip(y_pred, K.epsilon(), 1)
        # calc
        loss = y_true*K.log(y_pred)*self.weights
        loss =-K.sum(loss,-1)
        return loss
    

    


# test that it works that same as categorical_crossentropy with weights of one
import numpy as np
from keras.activations import softmax
from keras.objectives import categorical_crossentropy
import tensorflow as tf

samples=3
maxlen=4
vocab=5

sess = tf.InteractiveSession()
#tf.initialize_all_variables()

y_pred_n = np.random.random((samples,maxlen,vocab))
y_pred = K.variable(y_pred_n, name='y_pred_n')
y_true_n = softmax(tf.convert_to_tensor(np.random.random((samples,maxlen,vocab)))).eval()
y_true = K.variable(y_true_n, name='y_true') # this isn't binary
weights = np.ones(vocab)
weights_var = K.variable(weights, name='weights')
rr=categorical_crossentropy(y_true,y_pred)

zz = K.mean(K.square(y_pred - y_true), axis=-1)
#rr=categorical_crossentropy(y_true_n,y_pred_n)

init_op = tf.initialize_all_variables()
init_op.run()

print(zz.eval())

#y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
#        # clip
#y_pred = K.clip(y_pred, K.epsilon(), 1)
#        # calc
#loss = y_true*K.log(y_pred)*weights
#loss =-K.sum(loss,-1)
#
#loss2 = loss.eval()
#loss3 = rr.eval()
#print(loss2, loss3)
#
#
#    
##r=weighted_categorical_crossentropy(weights).eval()
##r=weighted_categorical_crossentropy(weights).loss(y_true_n,y_pred_n).eval()
##rr=categorical_crossentropy(tf.convert_to_tensor(y_true_n),tf.convert_to_tensor(y_pred_n)).eval()
##rr=categorical_crossentropy(y_true,y_pred).eval()
#np.testing.assert_almost_equal(loss2,loss3)
#print('OK')