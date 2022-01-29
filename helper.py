
from sklearn.metrics import classification_report, accuracy_score, balanced_accuracy_score
import numpy as np
import report
import flatten_json
from dataset import labels, label_names
import math
from torch.utils.data import DataLoader, TensorDataset, SubsetRandomSampler
from dataset import DeepCropDataset, ToTensor, labels as unique_labels, label_names
import torch

print = report.log

def output_size(n, k, p=0, s = 1):
  #[(W−K+2P)/S]+1
  #n = width and height dimensions 
  #k = kernel size 
  #p = padding 
  #s = stride
  return math.floor(((n + (2*p) - k) / s) + 1)

def output_size_3d(d_n, h_n, w_n, kernel_n, padding, stride = 1, dilation = 0): 
    
    d_out_len =  ((d_n + (2 * padding) - dilation * (kernel_n - 1) - 1) / stride) + 1
    h_out_len =  ((h_n + (2 * padding) - dilation * (kernel_n - 1) - 1) / stride) + 1
    w_out_len =  ((w_n + (2 * padding) - dilation * (kernel_n - 1) - 1) / stride) + 1
    return d_out_len, h_out_len, w_out_len

def evaluation(y_true, y_pred, initial=dict()): 
  #standard classfication report - precision, recall, f1-score, support

  res = classification_report(y_true, y_pred, output_dict=True, labels=range(len(labels)), target_names=label_names, zero_division=0)
  res["emr"] = emr(y_true, y_pred)
  res["one_zero_loss"] = one_zero_loss(y_true, y_pred)
  res["hamming_loss"] = hamming_loss(y_true, y_pred)
  res["accuracy"] = accuracy_score(y_true, y_pred)
  res = flatten_json.flatten(res)
  res_names = np.array(list(res.keys()))
  res_values = np.array(list(res.values()))
  res_names = np.concatenate((np.array(list(initial.keys())), res_names))
  res_values = np.concatenate((np.array(list(initial.values())), res_values))

  print("REPORT")
  print(classification_report(y_true, y_pred, labels=range(len(labels)), target_names=label_names, zero_division=0))
  print("MULTI-LABEL METRICS")
  print("EMR: {}".format(res["emr"]))
  print("1/0Loss: {}".format(res["one_zero_loss"]))
  print("Hamming Loss: {}".format(res["hamming_loss"]))
  return res_values, res_names

def scalars_from_scores(writer, scores, score_names, suffix=""):
  scores_mean = np.mean(scores, axis=0)
  for epoch, scores in enumerate(scores_mean):
    for score_i, score in enumerate(scores):
      writer.add_scalar(f'{score_names[score_i]}/{suffix}', score, epoch)

#Multi-Label Metrics    
def emr(y_true, y_pred):
  n = len(y_true)
  row_indicators = np.all(y_true == y_pred, axis = 1) # axis = 1 will check for equality along rows.
  exact_match_count = np.sum(row_indicators)
  return exact_match_count/n

def one_zero_loss(y_true, y_pred):
    n = len(y_true)
    row_indicators = np.logical_not(np.all(y_true == y_pred, axis = 1)) # axis = 1 will check for equality along rows.
    not_equal_count = np.sum(row_indicators)
    return not_equal_count/n

#Hamming Loss
#Hamming Loss computes the proportion of incorrectly predicted labels to the total number of labels.
def hamming_loss(y_true, y_pred):
    hl_num = np.sum(np.logical_xor(y_true, y_pred))
    hl_den = np.prod(y_true.shape)
    
    return hl_num/hl_den



def get_mean_std(times, batch_size = 1000):

    train_dataset = DeepCropDataset(csv_file="labels.csv", root_dir="data/prepared/train", times=times, transform=ToTensor())

    loader= DataLoader(train_dataset, batch_size=batch_size)
    ch_sum, ch_sqr_sum = 0, 0
    
    for batch in loader:
        data, labels = batch
        print(data.shape)
        print(type(data))
        ch_sum += torch.mean(data, dim = [0, 2, 3, 4])
        ch_sqr_sum += torch.mean(data**2, dim = [0, 2, 3, 4])
    
    mean = ch_sum/batch_size
    std = (ch_sqr_sum/batch_size - mean*2)*0.5
    
    return mean, std
