import keras
import numpy as np
from keras.layers import Conv2D,Input, MaxPool2D, AveragePooling2D, Dropout, Dense, Flatten
from keras.applications import VGG16
import tensorflow as tf
import os
from tensorflow.keras.initializers import RandomNormal as RN
from tensorflow.keras.initializers import GlorotNormal as GN
from tensorflow.keras.initializers import GlorotUniform as GU

def LR_model(lr, label = "abnormal",modelName = 'vgg'):
  METRICS = [
    tf.keras.metrics.TruePositives(name='tp'),
    tf.keras.metrics.FalsePositives(name='fp'),
    tf.keras.metrics.TrueNegatives(name='tn'),
    tf.keras.metrics.FalseNegatives(name='fn'), 
    tf.keras.metrics.BinaryAccuracy(name='accuracy'),
    tf.keras.metrics.Precision(name='precision'),
    tf.keras.metrics.Recall(name='recall'),
    tf.keras.metrics.AUC(name='auc'),
  ]

  model = keras.Sequential()
  model.add(Dense(1, activation="sigmoid", input_dim=3))
  model(Input(shape=(None, 3)))
  model.compile(
      optimizer=tf.keras.optimizers.Adam(lr=lr),
      loss=keras.losses.BinaryCrossentropy(),
      metrics=METRICS)
  data_path = "/content/gdrive/My Drive/Colab Notebooks/MRNet/"
  checkpoint_dir = data_path+"training_"+modelName+"_LR/" + label + "/"
  if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
  os.chdir(checkpoint_dir)
  cp_callback = keras.callbacks.ModelCheckpoint(checkpoint_dir+"weights.{epoch:02d}.hdf5",
                                                 save_weights_only=True,
                                                 verbose=1)
  return model, [cp_callback]










    