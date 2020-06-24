import os
import numpy as np
import glob
import csv

def partition(data_path):
  train_path = os.path.join(data_path, "train")
  test_path = os.path.join(data_path, "test")
  valid_path = os.path.join(data_path, "valid")
  sequences = ["axial", "coronal", "sagittal"]

  partition = {}
  partition["train"] = {}
  partition["valid"] = {}
  partition["test"] = {}

  for seq in sequences:
    seqPath_train = os.path.join(train_path, seq)
    seqPath_valid = os.path.join(valid_path, seq)
    seqPath_test = os.path.join(test_path, seq)
    os.chdir(seqPath_train)
    partition["train"][seq] = [os.path.splitext(f)[0] for f in glob.glob("*.npy")]
    os.chdir(seqPath_valid)
    partition["valid"][seq] = [os.path.splitext(f)[0] for f in glob.glob("*.npy")]
    os.chdir(seqPath_test)
    partition["test"][seq] = [os.path.splitext(f)[0] for f in glob.glob("*.npy")]

  labels = {}
  with open(f'{data_path}/train-abnormal.csv', newline='') as csvfile:
      data_train = list(csv.reader(csvfile))

  with open(f'{data_path}/valid-abnormal.csv', newline='') as csvfile:
      data_valid = list(csv.reader(csvfile))

  for item in data_train:
    labels[item[0]] = {"abnormal":int(item[1]), "ACL":0, "meniscus":0}

  for item in data_valid:
    labels[item[0]] = {"abnormal":int(item[1]), "ACL":0, "meniscus":0}

  with open(f'{data_path}/train-acl.csv', newline='') as csvfile:
      data_train = list(csv.reader(csvfile))

  with open(f'{data_path}/valid-acl.csv', newline='') as csvfile:
      data_valid = list(csv.reader(csvfile))

  for item in data_train:
    labels[item[0]]["ACL"] = int(item[1])

  for item in data_valid:
    labels[item[0]]["ACL"] = int(item[1])

  with open(f'{data_path}/train-meniscus.csv', newline='') as csvfile:
      data_train = list(csv.reader(csvfile))

  with open(f'{data_path}/valid-meniscus.csv', newline='') as csvfile:
      data_valid = list(csv.reader(csvfile))

  for item in data_train:
    labels[item[0]]["meniscus"] = int(item[1])

  for item in data_valid:
    labels[item[0]]["meniscus"] = int(item[1])
  return partition, labels