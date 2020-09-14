import time
import itertools
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import keras
from sklearn.model_selection import train_test_split
# %matplotlib inline
# tf.__version__

def f2(x):
    '''
    Funcao não linear a ser aprendida
    '''
    return (x**2 + x*3 + 4)/200


def train(_optimizer="adam", _units=2, _activation_l1="relu", _activation_l2="relu"):
    """
    Possiveis ativacoes: ['elu','exponential','hard_sigmoid','linear','relu','selu','sigmoid','softmax','softplus','softsign','swish','tanh']
    Possiveis otimizadores: ['SGD', 'RMSprop', 'Adam', 'Adadelta', 'Adagrad', 'Adamax', 'Nadam', 'Ftrl']
    """
    x = np.linspace(0,10,100)
    y = f2(x)

    model = tf.keras.Sequential([ 
        keras.Input(shape=(1,)),
        # Seu código aqui
        keras.layers.Dense(units=_units, activation=_activation_l1),
        # keras.layers.Dense(units=_units, activation=_activation_l2),
        keras.layers.Dense(1),
    ])

    model.compile(optimizer=_optimizer, loss="mean_squared_error")
    model.fit(x,y,epochs=400)
    return model




# activation = "relu"
# units =2 

# x_val = np.linspace(0,10,63)
x_val = np.linspace(20,30,63)
y_val = f2(x_val)

# a = ['relu', 'sigmoid', 'softmax', 'softplus', 'softsign', 'tanh', 'selu', 'elu', 'exponential']
# a = ['sigmoid','softmax','softplus','softsign','swish','tanh']
# a = ['linear']
# a = ['elu','exponential','hard_sigmoid','linear','relu','selu','sigmoid','softmax','softplus','softsign','swish','tanh']
# b = [2,3,4,5,6]

a = ['relu']
b = [2]
optimizer = "SGD"
# Duas camadas
# for combination in itertools.product(a,a,b):
# 	activation_l1 = combination[0]
# 	activation_l2 = combination[1]
# 	units = combination[2]
	# model = train(units, activation_l1, activation_l2)
	# print(f"##Ativacao1:#{activation_l1}#Ativacao2:#{activation_l2}#Nodes:#{units}#Loss: #{loss}#Time:#{elapsed}")


for combination in itertools.product(a,b):
	activation_l1 = combination[0]
	units = combination[1]
	start = time.time()
	model = train(optimizer, units, activation_l1 )
	elapsed = time.time() - start
	loss = model.evaluate(x=x_val,y=y_val)

	print(f"##Ativacao:#{activation_l1}#Nodes:#{units}#Loss: #{loss}#Time:#{elapsed}")
	# print(f"##Ativacao1:#{activation_l1}#Ativacao2:#{activation_l2}#Nodes:#{units}#Loss: #{loss}#Time:#{elapsed}")

