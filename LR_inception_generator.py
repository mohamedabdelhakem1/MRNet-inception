import keras
import numpy as np
import os
from PIL import Image
import random, copy

MAX_PIXEL_VAL = 255
MEAN = 58.09
STDDEV = 49.73

class MRNet_data_generator(keras.utils.Sequence):
  def __init__(self, datapath, IDs, labels, batch_size = 1, 
               scale_to = (299, 299), label_type="abnormal", exam_type="axial",
               data_type='train'):
    print("Initializing Data Generator:")
    self.path = datapath
    self.n = 0
    self.original_IDs = IDs
    self.IDs = IDs
    self.labels = labels
    self.batch_size = batch_size
    self.scale_to = scale_to
    self.label_type = label_type
    self.exam_type = exam_type
    self.current_exam = None
   
    
    self.temp_n = 0
    self.data_type = data_type
    self.data_path = os.path.join(self.path, self.data_type)
    print("data type: ", self.data_type)
    print("Combination: ", self.label_type, " and " , self.exam_type)
    print("data path: ", self.data_path)
    self.on_epoch_end()
    self.end = self.__len__()
    print("Number of inputs: ", self.end)
    print("input size: ", self.scale_to)
    self.IDs[self.data_type][self.exam_type] = sorted(self.IDs[self.data_type][self.exam_type])



  def on_epoch_end(self):
   return

  def __data_generation(self, list_IDs_temp):
    'Generates data containing batch_size samples' 
    y = np.empty((self.batch_size), dtype=int)
    arr = []
    for i, ID in enumerate(list_IDs_temp):
        exam_path = os.path.join(self.data_path, self.exam_type)
        exam = np.load(os.path.join(exam_path, ID+'.npy'))
        e = []
        for s in exam:
          im = Image.fromarray(s)
          s = np.array(im.resize(self.scale_to), dtype=np.float32)
          # standardize
          s = (s - np.min(s)) / (np.max(s) - np.min(s)) * MAX_PIXEL_VAL
          # normalize
          s = (s - MEAN) / STDDEV
          expanded = np.array([s])
          e.append(expanded.reshape((self.scale_to[0], self.scale_to[1], 1)))

        e = np.array(e)
        arr.append(e)
        y[i] = self.labels[ID][self.label_type]
        
    X = np.array(arr)
    return X, y

  def __len__(self):
    'Denotes the number of batches per epoch'
    IDs_len = len(self.IDs[self.data_type][self.exam_type])
    return int(np.floor(IDs_len / self.batch_size))

  def __getitem__(self, index):
    'Generate one batch of data'
    list_IDs_temp = [self.IDs[self.data_type][self.exam_type][index]]
    X, y = self.__data_generation(list_IDs_temp)
    X, y = self.expand_channels(X[0], y[0], 0)
    self.current_exam = (X, y)
    return X, y

  def expand_channels(self, exam, label, batch_size=1):
    augmented_batch = []
    augmented_batch_labels = []
    e = []
    for s in range(0, exam.shape[0]):
      scan = exam[s]
      scan = scan.reshape((self.scale_to[0], self.scale_to[1]))
      scan = np.array([scan, scan, scan]).reshape((self.scale_to[0], self.scale_to[1], 3))
      e.append(scan)
    augmented_batch.append(e)
    augmented_batch_labels.append(label)
    return np.array(augmented_batch), np.array(augmented_batch_labels)


  def __next__(self):
    if self.n >= self.end:
      self.n = 0
    result = self.__getitem__(self.n)
    self.n += 1
    return result
