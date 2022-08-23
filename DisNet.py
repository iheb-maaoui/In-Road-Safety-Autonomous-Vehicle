from distutils.command.install_egg_info import to_filename


import tensorflow as tf
model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(6,)),
    tf.keras.layers.Dense(100,'relu',kernel_initializer='normal'),
    tf.keras.layers.Dense(100,'relu',kernel_initializer='normal'),
    tf.keras.layers.Dense(100,'relu',kernel_initializer='normal'),
    tf.keras.layers.Dense(100,'relu',kernel_initializer='normal'),
    tf.keras.layers.Dense(100,'relu',kernel_initializer='normal'),
    tf.keras.layers.Dense(100,'relu',kernel_initializer='normal'),
    tf.keras.layers.Dense(1,'relu')
])

from keras import backend as K

def RSquare(y_true, y_pred):
    SS_res =  K.sum(K.square( y_true-y_pred ))
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )

from keras.optimizers import Adam

model.compile(loss='mse',optimizer=Adam(learning_rate=0.00001),metrics=[RSquare])