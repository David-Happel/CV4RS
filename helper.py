
from sklearn.metrics import classification_report, accuracy_score, balanced_accuracy_score
import numpy as np
import report
import flatten_json

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
  label_names = ['Grassland', 'Winter Wheat', 'Winter Rye', 'Winter Barley', 'Other Winter Cereals', 'Spring Barley', 'Spring Oat', 'Other Spring Cereals', 'Winter Rapeseed', 'Legume', 'Sunflower',
                  'Sugar Beet', 'Maize other', 'Maize for grain', 'Potato', 'Strawberry', 'Asparagus', 'Onion', 'Carrot', 'Other leafy Vegetables']
  labels = [10, 31, 32, 33, 34, 41, 42, 43, 50, 60, 70, 80, 91, 92, 100, 120, 130, 140, 181, 182]
  
  return labels, label_names


def evaluation(y_true, y_pred, initial=dict()): 
  labels, label_names = get_labels()
  #standard classfication report - precision, recall, f1-score, support

  res = classification_report(y_true, y_pred, output_dict=True, labels=range(len(get_labels()[0])), target_names=get_labels()[1], zero_division=0)
  res["emr"] = emr(y_true, y_pred)
  res["one_zero_loss"] = one_zero_loss(y_true, y_pred)
  res["hamming_loss"] = hamming_loss(y_true, y_pred)
  res["accuracy"] = accuracy_score(y_true, y_pred)
  res = flatten_json.flatten(res)
  res_names = np.array(list(res.keys()))
  res_values = np.array(list(res.values()))
  res_names = np.concatenate((np.array(list(initial.keys())), res_names))
  res_values = np.concatenate((np.array(list(initial.values())), res_values))


  # print(classification_report(y_true, y_pred, labels=range(len(get_labels()[0])), target_names=get_labels()[1], zero_division=0))
  # print("MULTI-LABEL METRICS")
  # print("EMR: {}".format(res["emr"]))
  # print("1/0Loss: {}".format(res["one_zero_loss"]))
  # print("Hamming Loss: {}".format(res["hamming_loss"]))
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

