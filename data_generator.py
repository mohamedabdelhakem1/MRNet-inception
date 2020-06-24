import keras
import numpy as np
import os
from PIL import Image
import random, copy

MAX_PIXEL_VAL = 255
MEAN = 58.09
STDDEV = 49.73

class MRNet_data_generator(keras.utils.Sequence):
  def __init__(self, datapath, IDs, labels,class_count=None, batch_size = 1, shuffle=True,
               scale_to = (256, 256), label_type="abnormal", exam_type="axial",
               data_type='train', model="vgg", aug_size=1, keep_original=1):
    print("Initializing Data Generator:")
    self.path = datapath
    self.n = 0
    self.original_IDs = IDs
    self.IDs = IDs
    self.labels = labels
    self.batch_size = batch_size
    self.shuffle = shuffle
    self.scale_to = scale_to
    self.label_type = label_type
    self.exam_type = exam_type
    self.model = model
    self.current_exam = None
    self.class_count=class_count
    self.keep_original = keep_original
    if class_count is None:
      self.factor = 1
    elif class_count[0] > class_count[1]:
      self.factor = class_count[0] / class_count[1]
      self.repeat = 1
    else:
      self.factor = class_count[1] / class_count[0]
      self.repeat = 0
    
    self.temp_n = 0
    print('factor : ', self.factor)
    self.aug_size=aug_size
    self.data_type = data_type
    self.data_path = os.path.join(self.path, self.data_type)
    print("model: ", self.model)
    print("data type: ", self.data_type)
    print("Combination: ", self.label_type, " and " , self.exam_type)
    print("data path: ", self.data_path)
    self.on_epoch_end()
    self.end = self.__len__()
    print("Number of inputs: ", self.end)
    print("input size: ", self.scale_to)
    

  def _repeat_exams(self):
    self.IDs = copy.deepcopy(self.original_IDs)
    if self.data_type == 'valid':
      return
    if self.factor == 1:
      return
    f = self.factor - 1
    repeated = int(f * self.class_count[self.repeat])
    keys = []
    for k in self.IDs[self.data_type][self.exam_type]:
      if self.repeat == self.labels[k][self.label_type]:
        keys.append(k)
    keys = random.choices(keys, k=repeated)
    self.IDs[self.data_type][self.exam_type] += keys    

  def on_epoch_end(self):
    'Updates indexes after each epoch'
    self._repeat_exams()
    self.indexes = np.arange(len(self.IDs[self.data_type][self.exam_type]))
    if self.shuffle == True:
        np.random.shuffle(self.indexes)

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
    indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
    list_IDs_temp = [self.IDs[self.data_type][self.exam_type][k] for k in indexes]
    X, y = self.__data_generation(list_IDs_temp)
    X, y = self.augment_data(X[0], y[0], batch_size=self.aug_size)
    self.current_exam = (X, y)
    return X, y

  def augment_data(self, exam, label, batch_size=1, use_random_rotation=True, use_random_shear=False, use_random_shift=True, use_random_flip=True):
    # if label == 0 and self.class_weight[0] > self.class_weight[1]:
    #   batch_size = int((self.class_weight[0]/self.class_weight[1])*(batch_size+1))
    # elif label == 1 and self.class_weight[0] < self.class_weight[1]:
    #   batch_size = int((self.class_weight[1]/self.class_weight[0])*(batch_size+1))
    augmented_batch = []
    augmented_batch_labels = []
    if self.keep_original:
      e = []
      for s in range(0, exam.shape[0]):
        scan = exam[s]
        scan = scan.reshape((self.scale_to[0], self.scale_to[1]))
        scan = np.array([scan, scan, scan]).reshape((self.scale_to[0], self.scale_to[1], 3))
        e.append(scan)
    
      augmented_batch.append(e)
      augmented_batch_labels.append(label)
    else:
      batch_size += 1
    for i in range (0, batch_size):
      e = []
      for s in range(0, exam.shape[0]):
        scan = exam[s]
        if use_random_rotation:
          scan = keras.preprocessing.image.random_rotation(scan, 25, row_axis=1, col_axis=2, channel_axis=0)
        if use_random_shear:
          scan = keras.preprocessing.image.random_shear(scan, 0.2, row_axis=1, col_axis=2, channel_axis=0)
        if use_random_shift:
          rg = float(25/scan.shape[1])
          scan = keras.preprocessing.image.random_shift(scan, rg, rg, row_axis=1, col_axis=2, channel_axis=0)
        if use_random_flip:
          if bool(random.getrandbits(1)):
            scan = np.fliplr(scan)
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
