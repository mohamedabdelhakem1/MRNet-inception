import keras
import tensorflow as tf
import numpy as np 
from keras.regularizers import l2
from keras import Model
from keras.layers import Conv2D,BatchNormalization, MaxPool2D, Dropout, Dense, Input, concatenate, GlobalAveragePooling2D, AveragePooling2D, Flatten,Activation
import os
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import auc as roc_auc


class inceptionV3():
  def __init__(self,shape ,weightsPath = None):
    self.shape =shape;
    self.weightsPath = weightsPath;

  def ConvBatchNorm(self,filters,kernel_size,padding,strides=(1,1)):
    def inp(input):
      x = Conv2D(filters=filters,kernel_size=kernel_size,strides=strides,padding=padding,use_bias=False)(input);
      x = BatchNormalization(axis=3,scale=False)(x)
      x = Activation('relu')(x)
      return x
    return inp;

  def inceptionModlueA(self ,inp,filter1_1x1,filter2_pool,filter3_1x1,filter3_5x5,filter4_1x1,filter4_3x3,filter4_3x3_2,kernel_init="glorot_uniform",
    bias_init="zeros",name=None):
    conv1  = self.ConvBatchNorm(filters=filter1_1x1,kernel_size=(1,1),padding='same')(inp)
  
    conv2_1  = AveragePooling2D((3,3),strides=(1,1),padding='same')(inp)
    conv2_2 =  self.ConvBatchNorm(filters=filter2_pool,kernel_size=(1,1),padding='same')(conv2_1)
    
    conv3_1 = self.ConvBatchNorm(filters=filter3_1x1,kernel_size=(1,1),padding='same')(inp)
    conv3_2 = self.ConvBatchNorm(filters=filter3_5x5,kernel_size=(5,5),padding='same')(conv3_1)
    
    conv4_1 = self.ConvBatchNorm(filters=filter4_1x1 , kernel_size=(1,1),padding='same')(inp)
    conv4_2 = self.ConvBatchNorm(filters= filter4_3x3 ,kernel_size=(3,3),padding='same')(conv4_1)
    conv4_3 = self.ConvBatchNorm(filters= filter4_3x3_2 ,kernel_size=(3,3),padding='same')(conv4_2)
    
    output = concatenate([conv1, conv2_2, conv3_2, conv4_3], axis=3, name=name)
    return output

  def inceptionModuleB(self,inp,filter1_1x1,filter2_pool,filter3_1x1,filter3_1xn,filter3_nx1,filter4_1x1,filter4_1xn,filter4_nx1,filter4_1xn_1,filter4_nx1_2,kernel_init="glorot_uniform",
    bias_init="zeros",name=None):
    conv1  = self.ConvBatchNorm(filters=filter1_1x1,kernel_size=(1,1),padding='same')(inp)

    conv2_1  = AveragePooling2D((3,3),strides=(1,1),padding='same')(inp)
    conv2_2 =  self.ConvBatchNorm(filters=filter2_pool,kernel_size=(1,1),padding='same')(conv2_1)
    
    conv3_1 = self.ConvBatchNorm(filters=filter3_1x1,kernel_size=(1,1),padding='same')(inp)
    conv3_2 = self.ConvBatchNorm(filters=filter3_1xn,kernel_size=(1,7),padding='same')(conv3_1)
    conv3_3 = self.ConvBatchNorm(filters=filter3_nx1,kernel_size=(7,1),padding='same')(conv3_2)
    
    conv4_1 = self.ConvBatchNorm(filters=filter4_1x1,kernel_size=(1,1),padding='same')(inp)
    conv4_2 = self.ConvBatchNorm(filters=filter4_1xn,kernel_size=(1,7),padding='same')(conv4_1)
    conv4_3 = self.ConvBatchNorm(filters=filter4_nx1,kernel_size=(7,1),padding='same')(conv4_2)
    conv4_4 = self.ConvBatchNorm(filters=filter4_1xn_1,kernel_size=(1,7),padding='same')(conv4_3)
    conv4_5 = self.ConvBatchNorm(filters=filter4_nx1_2,kernel_size=(7,1),padding='same')(conv4_4)
    
    output = concatenate([conv1, conv2_2, conv3_3, conv4_5], axis=3, name=name)
    return output
    
  def inceptionModuleC(self,inp,filter1_1x1,filter2_1x1,filter3_1x1,filter3_1x3,filter3_3x1,filter4_1x1,filter4_3x3,filter4_1x3,filter4_3x1,kernel_init="glorot_uniform",
    bias_init="zeros",name=None):
    conv1  = self.ConvBatchNorm(filters=filter1_1x1,kernel_size=(1,1),padding='same')(inp)

    conv2_1  = AveragePooling2D((3,3),strides=(1,1),padding='same')(inp)
    conv2_2 =  self.ConvBatchNorm(filters=filter2_1x1,kernel_size=(1,1),padding='same')(conv2_1)
    
    conv3_1 = self.ConvBatchNorm(filters=filter3_1x1,kernel_size=(1,1),padding='same')(inp)
    conv3_2 = self.ConvBatchNorm(filters=filter3_1x3,kernel_size=(1,3),padding='same')(conv3_1)
    conv3_3 = self.ConvBatchNorm(filters=filter3_3x1,kernel_size=(3,1),padding='same')(conv3_1)
    conv_3 = concatenate([conv3_2,conv3_3],axis=3);
    
    conv4_1 = self.ConvBatchNorm(filters=filter4_1x1 , kernel_size=(1,1),padding='same')(inp)
    conv4_2 = self.ConvBatchNorm(filters= filter4_3x3 ,kernel_size=(3,3),padding='same')(conv4_1)
    conv4_3 = self.ConvBatchNorm(filters= filter4_1x3 ,kernel_size=(1,3),padding='same')(conv4_2)
    conv4_4 = self.ConvBatchNorm(filters= filter4_3x1 ,kernel_size=(3,1),padding='same')(conv4_2)
    conv_4 = concatenate([conv4_3,conv4_4],axis=3);

    output = concatenate([conv1, conv2_2,conv_3,conv_4], axis=3, name=name)
    return output

  def inceptionModlueD(self,x,filter1_3x3 , filter2_1x1, filter2_3x3 ,  filter2_3x3_2, name=None):
    conv1 = self.ConvBatchNorm(filters=filter1_3x3, kernel_size=(3,3),strides=(2,2),padding='valid')(x);

    conv2_1 = self.ConvBatchNorm(filters=filter2_1x1, kernel_size=(1,1),padding='same')(x);
    conv2_2 = self.ConvBatchNorm(filters=filter2_3x3, kernel_size=(3,3),padding='same')(conv2_1);
    conv2_3 = self.ConvBatchNorm(filters=filter2_3x3_2, kernel_size=(3,3),strides=(2,2),padding='valid')(conv2_2);

    conv3 = MaxPool2D(pool_size=(3,3) , strides=(2,2),padding='valid')(x);

    output = concatenate([conv1, conv2_3,conv3], axis=3, name=name)
    return output
  def inceptionModlueE(self,x,filter1_1x1,filter1_3x3 , filter2_1x1, filter2_1x7 ,filter2_7x1,filter2_3x3,name=None):
    conv1_1 = self.ConvBatchNorm(filters=filter1_1x1,kernel_size=(1,1),padding='same')(x);
    conv1_2 = self.ConvBatchNorm(filters=filter1_3x3,kernel_size=(3,3),strides=(2,2),padding='valid')(conv1_1);
    
    conv2_1 =  self.ConvBatchNorm(filters=filter2_1x1,kernel_size=(1,1),padding='same')(x);
    conv2_2 =  self.ConvBatchNorm(filters=filter2_1x7,kernel_size=(1,7),padding='same')(conv2_1);
    conv2_3 =  self.ConvBatchNorm(filters=filter2_7x1,kernel_size=(7,1),padding='same')(conv2_2);
    conv2_4 =  self.ConvBatchNorm(filters=filter2_3x3,kernel_size=(3,3),strides=(2,2),padding='valid')(conv2_3);
    
    conv3 = MaxPool2D(pool_size=(3,3),strides=2,padding='valid')(x);
    output = concatenate([conv1_2, conv2_4,conv3], axis=3, name=name)
    return output

  def getModel(self):
    input = Input(shape=self.shape);
    # 224x224x3
    x = self.ConvBatchNorm(filters=32,kernel_size=(3,3),strides=(2,2),padding='valid')(input)
    # 112x112x32
    x = self.ConvBatchNorm(filters=32,kernel_size=(3,3),strides=(1,1),padding='valid')(x)
    # 110x110x32
    x = self.ConvBatchNorm(filters=64,kernel_size=(3,3),strides=(1,1),padding='same')(x)  
    # 110x110x64
    x = MaxPool2D(pool_size=(3,3),strides=(2,2))(x)
    # 55x55x64
    x = self.ConvBatchNorm(filters=80,kernel_size=(1,1),strides=(1,1),padding='valid')(x)
    # 55x55x80
    x = self.ConvBatchNorm(filters=192,kernel_size=(3,3),strides=(1,1),padding='valid')(x)
    # 53x53x192
    x = MaxPool2D(pool_size=(3,3),strides=(2,2))(x)
    # 26x26x192
    x = self.inceptionModlueA(x,filter1_1x1=64,filter2_pool=32,filter3_1x1=48,filter3_5x5=64,filter4_1x1=64,filter4_3x3=96,filter4_3x3_2=96);
    # 26x26x256
    x = self.inceptionModlueA(x,filter1_1x1=64,filter2_pool=64,filter3_1x1=48,filter3_5x5=64,filter4_1x1=64,filter4_3x3=96,filter4_3x3_2=96);
    # 26x26x288
    x = self.inceptionModlueA(x,filter1_1x1=64,filter2_pool=64,filter3_1x1=48,filter3_5x5=64,filter4_1x1=64,filter4_3x3=96,filter4_3x3_2=96);
    # 26x26x288

    x = self.inceptionModlueD(x,filter1_3x3=384 , filter2_1x1=64, filter2_3x3=96 ,  filter2_3x3_2=96);
    # 13x13x768

    x =  self.inceptionModuleB(x,filter1_1x1=192,filter2_pool=192,filter3_1x1=128,filter3_1xn=128,filter3_nx1=192,filter4_1x1=128,filter4_1xn=128,filter4_nx1=128
                          ,filter4_1xn_1=128,filter4_nx1_2=192);
    x =  self.inceptionModuleB(x,filter1_1x1=192,filter2_pool=192,filter3_1x1=160,filter3_1xn=160,filter3_nx1=192,filter4_1x1=160,filter4_1xn=160,filter4_nx1=160
                          ,filter4_1xn_1=160,filter4_nx1_2=192);
                          
    x =  self.inceptionModuleB(x,filter1_1x1=192,filter2_pool=192,filter3_1x1=160,filter3_1xn=160,filter3_nx1=192,filter4_1x1=160,filter4_1xn=160,filter4_nx1=160
                          ,filter4_1xn_1=160,filter4_nx1_2=192);

    x =  self.inceptionModuleB(x,filter1_1x1=192,filter2_pool=192,filter3_1x1=192,filter3_1xn=192,filter3_nx1=192,filter4_1x1=192,filter4_1xn=192,filter4_nx1=192
                          ,filter4_1xn_1=192,filter4_nx1_2=192);

    x = self.inceptionModlueE(x,filter1_1x1=192,filter1_3x3=320 , filter2_1x1=192, filter2_1x7=192 ,filter2_7x1 =192 ,filter2_3x3=192);

    x = self.inceptionModuleC(x,filter1_1x1=320,filter2_1x1=192,filter3_1x1=384,filter3_1x3=384,filter3_3x1=384,filter4_1x1=448,filter4_3x3=384,filter4_1x3=384,filter4_3x1=384);

    x = self.inceptionModuleC(x,filter1_1x1=320,filter2_1x1=192,filter3_1x1=384,filter3_1x3=384,filter3_3x1=384,filter4_1x1=448,filter4_3x3=384,filter4_1x3=384,filter4_3x1=384);

    model = Model(inputs=input, outputs =x, name='inception_v3')
    # model.summary()
    
    if self.weightsPath:
        model.load_weights(self.weightsPath)
    return model

class MRNet_inception_layer(keras.layers.Layer):
  def __init__(self, batch_size):
    super(MRNet_inception_layer, self).__init__()
    self.inception = inceptionV3((299, 299, 3)).getModel()
    self.avg_pooling1 = AveragePooling2D(pool_size=(8, 8), padding="same")
    # self.d1 = Dropout(0.1)
    self.fc1 = Dense(1, activation="sigmoid", input_dim=2048)
    self.b_size = batch_size

  def compute_output_shape(self, input_shape):
    return (None, 1)
  
  def call(self, inputs):
    arr1 = []
    for index in range(self.b_size):
     
      f_list = self.inception(inputs[index])
      out1 = tf.squeeze(self.avg_pooling1(f_list), axis=[1, 2])
      out1 = keras.backend.max(out1, axis=0, keepdims=True)
      out1 = tf.squeeze(out1)
      arr1.append(out1)
    print("no dropout")
    out1 = tf.stack(arr1, axis=0)
    out1 = self.fc1(out1)
    return out1


def MRNet_inc_model(batch_size,lr ,combination = ["abnormal", "axial"]):
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
  b_size = batch_size
  model = keras.Sequential()
  model.add(MRNet_inception_layer(b_size))
  model(Input(shape=(None ,299, 299, 3)))
  model.summary()

  model.compile(
   optimizer=keras.optimizers.Adam(learning_rate=lr)
    ,   loss=keras.losses.BinaryCrossentropy(label_smoothing=0.1),
      metrics=METRICS)

  data_path = "/content/gdrive/My Drive/Colab Notebooks/MRNet/"
  checkpoint_dir = data_path+"training_inception/" + combination[0] + "/" + combination[1] + "/"
  # checkpoint_dir = os.path.dirname(checkpoint_path)
  if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
  os.chdir(checkpoint_dir)
  cp_callback = keras.callbacks.ModelCheckpoint(checkpoint_dir+"weights.{epoch:02d}.hdf5",
                                                 save_weights_only=True,
                                                 verbose=1)

  return model, [cp_callback,validation_Callback]
  
class validation_Callback(tf.keras.callbacks.Callback):
  def __init__(self, model, valid_data_gen):
    super(validation_Callback, self).__init__()
    self.model = model
    self.valid_data_gen = valid_data_gen

  def on_epoch_end(self, epoch, logs=None):
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    y_true = []
    y_score = []
    for i in range(len(self.valid_data_gen)):
      x, y = next(self.valid_data_gen)
      _y = self.model.predict_classes(x, batch_size=1, verbose=0)
      _y_score = self.model.predict_proba(x, batch_size=1)
      y_score.append(_y_score[0][0])
      y_true.append(y[0])
      if _y[0][0] == 1:
        if _y[0][0] == y[0]:
          tp += 1
        else:
          fp += 1
      else:
        if _y[0][0] == y[0]:
          tn += 1
        else:
          fn += 1
    y_score = np.array(y_score)
    y_true = np.array(y_true)
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc(fpr, tpr)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2*((precision*recall)/(precision+recall))
    print ("\n validation: tp = ", tp, " fp = ", fp, " tn = ", tn, " fn = ", fn, " accuracy = ", (tp+tn)/(len(valid_data_gen)), " auc = ", auc, "F1 Score = ", f1, "\n")

    

