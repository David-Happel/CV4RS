
from sklearn.metrics import classification_report
import numpy as np
import report

print = report.log

def output_size(d_n, h_n, w_n, kernel_n, padding, stride = 1, dilation = 0): 
    
    d_out_len =  ((d_n + (2 * padding) - dilation * (kernel_n - 1) - 1) / stride) + 1
    h_out_len =  ((h_n + (2 * padding) - dilation * (kernel_n - 1) - 1) / stride) + 1
    w_out_len =  ((w_n + (2 * padding) - dilation * (kernel_n - 1) - 1) / stride) + 1
    return d_out_len, h_out_len, w_out_len


def reset_weights(m):
  '''
    Try resetting model weights to avoid
    weight leakage.
  '''
  for layer in m.children():
   if hasattr(layer, 'reset_parameters'):

    # print(f'Reset trainable parameters of layer = {layer}')
    layer.reset_parameters()

def get_labels(): 
  label_names = ['No class', 'Grassland', 'Winter Wheat', 'Winter Rye', 'Winter Barley', 'Other Winter Cereals', 'Spring Barley', 'Spring Oat', 'Other Spring Cereals', 'Winter Rapeseed', 'Legume', 'Sunflower',
                  'Sugar Beet', 'Maize other', 'Maize for grain', 'Potato', 'Grapevine', 'Strawberry', 'Asparagus', 'Onion', 'Hops', 'Orchard', 'Carrot', 'Other leafy Vegetables']
  labels = [0, 10, 31, 32, 33, 34, 41, 42, 43, 50, 60, 70, 80, 91, 92, 100, 110, 120, 130, 140, 150, 160, 181, 182]
  
  return labels, label_names


def evaluation(y_true, y_pred): 
  labels, label_names = get_labels()
  #standard classfication report - precision, recall, f1-score, support

  # print(classification_report(y_true, y_pred, target_names=label_names))
  res = classification_report(y_true, y_pred, output_dict=True, labels=range(len(get_labels()[0])), target_names=get_labels()[1], zero_division=0)
  res["emr"] = emr(y_true, y_pred)
  res["one_zero_loss"] = one_zero_loss(y_true, y_pred)
  res["hamming_loss"] = hamming_loss(y_true, y_pred)

  print(classification_report(y_true, y_pred, labels=range(len(get_labels()[0])), target_names=get_labels()[1], zero_division=0))
  print("MULTI-LABEL METRICS")
  print("EMR: {}".format(res["emr"]))
  print("1/0Loss: {}".format(res["one_zero_loss"]))
  print("Hamming Loss: {}".format(res["hamming_loss"]))
  return res

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

