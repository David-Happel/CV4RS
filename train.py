from enum import unique
import os
import json
import numpy as np
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, SubsetRandomSampler
from torchsummary import summary

from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, accuracy_score

from baseline_simple import C3D as bl
from processdata import ProcessData
from helper import reset_weights, get_labels, evaluation
import report

print = report.log

### CONFIG
device = t.device('cuda' if t.cuda.is_available() else 'cpu')

if t.cuda.is_available():
    print(f'\nUsing gpu {t.cuda.current_device()}')
else:
    print(f'\nUsing cpu')

os.chdir(os.path.dirname(os.path.abspath(__file__)))
print("Working dir: " + os.getcwd())

# Change if need to process the data
process_data = False

#create band and times arrays
t_start = 1
t_stop = 37
t_step = 6
times = range(t_start,t_stop,t_step)
bands = ["GRN", "NIR", "RED"]
labels, label_names = get_labels()

#Restriction of samples to take
t_samples = 10

class_weights = True

### Main Function
def main():
    #PREPARE DATA 
    dl = ProcessData(bands = bands, times=times)
    if process_data:
        print("Preparing data")
        #process training data 
        dl.process_tile("X0071_Y0043", out_dir = 'data/prepared/train/')
        #process test data 
        dl.process_tile("X0071_Y0043", out_dir='data/prepared/test/')

    #create dataset
    #data format (sample, band, time, height, width)
    print("Loading data")
    train_data, train_labels = dl.read_dataset(out_dir='data/prepared/train/', t_samples=t_samples)
    #converting to tensor datasets
    train_ds = TensorDataset(train_data , train_labels)

    labels_n = train_labels.shape[1]
    print("Labels: " + str(labels_n))

    # print(train_labels)
    if class_weights:
        pos_weights = train_labels.shape[0] / t.sum(train_labels, axis=0)
        pos_weights[pos_weights == float('inf')] = 1
    else: pos_weights = np.ones((train_labels.shape[0]))
    print(f'Pos Weights: {pos_weights}')

    #TRAINING
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)

    epochs = 3
    k_folds = 5

    kfold = KFold(n_splits=k_folds, shuffle=True)

    train_scores = np.array([])
    val_scores =  np.array([])
    for fold, (train_ids, test_ids) in enumerate(kfold.split(train_ds)):

        print(f'======= FOLD: {fold+1} ==================================')
        # Sample elements randomly from a given list of ids, no replacement.
        train_subsampler = SubsetRandomSampler(train_ids)
        test_subsampler = SubsetRandomSampler(test_ids)

        # Define data loaders for training and testing data in this fold
        train_batches = DataLoader(
                        train_ds, 
                        batch_size=5, sampler=train_subsampler)
        test_batches = DataLoader(
                        train_ds,
                        batch_size=5, sampler=test_subsampler)

        #model selection

        model = bl(bands=len(bands), labels=labels_n).to(device)
        # model.apply(reset_weights)

        # optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        optimizer = optim.Adam(model.parameters(), lr = 0.001)

        best_f1 = 0

        for epoch in range(epochs):  # loop over the dataset multiple times
            print(f'---- EPOCH: {epoch+1} -------------------------------')
            # TRAIN EPOCH
            train_score = train(model, train_batches, device=device, optimizer=optimizer, criterion=criterion)
            train_scores = np.append(train_scores, [train_score])
            
            # TEST EPOCH
            test_score = test(model, test_batches, device=device, criterion=criterion)
            val_scores = np.append(val_scores, [test_score])
            
            if(test_score["weighted avg"]["f1-score"] > best_f1):
                save_path = f'{report.report_dir}/saved_model/model-fold-{fold+1}.pth'
                t.save(model.state_dict(), save_path)
                best_f1 = test_score["weighted avg"]["f1-score"]
                print(f'Saved Epoch model for {epoch+1}')
            
            
            
    with open(f'{report.report_dir}/train_scores.json', 'w') as fp:
        json.dump(train_scores.tolist(), fp)
    with open(f'{report.report_dir}/val_scores.json', 'w') as fp:
        json.dump(val_scores.tolist(), fp)

    # Print fold results
    print(f'K-FOLD CROSS VALIDATION RESULTS FOR {k_folds} FOLDS')
    loss_sum = sum(map(lambda fold: fold["loss"],val_scores))
    print(f'Average Loss: {loss_sum/len(val_scores)}')

    f1_sum = sum(map(lambda fold: fold['weighted avg']["f1-score"],val_scores))
    print(f'Average f1: {f1_sum/len(val_scores)}')



    #TEST EVALUATION
    ## #TODO  work on test set

    # test_data, test_labels = dl.read_dataset(out_dir='data/prepared/test/')
    # test_ds = TensorDataset(test_data , test_labels)



### Training Functions

# train the model
def train(model, batches, device="cpu", optimizer = None, criterion = None):
    model.train()

    avg_loss = 0
    # TODO: fix for actual label count
    y_pred = np.empty((0, 19))
    y_true = np.empty((0, 19))

    for batch_i, batch in enumerate(batches):
        print('Batch: {}/{}'.format(batch_i+1, len(batches)))
        
        # batch is a list of 10 [inputs, labels]  -> e.g. len(labels) = 10
        inputs, labels = batch
        inputs, labels = inputs.to(device), labels.to(device)
        
        # improve params. for the batch
        optimizer.zero_grad()
        
        # forward + backward + optimize  for the batch
        # outputs =  logits, probs ) probabilities from sigmoid layer
        outputs, probs = model(inputs)
        #print('outputs', outputs)
        #print('probs', probs)
        
        # compute loss using logits
        loss = criterion(outputs, labels)
            
        # do backprop. for the batch
        loss.backward()

        # update step for the batch
        optimizer.step()
            
        # predicted labels for F1 score
        predicted = t.round(probs).to(device)
        #print('predicted:', predicted)

        avg_loss += loss.item()

        y_pred =  np.append(y_pred, predicted.detach().to('cpu'), axis=0)
        y_true =  np.append(y_true, labels.detach().to('cpu'), axis=0)
        
    res = evaluation(y_pred, y_true)
    res["loss"] = avg_loss / len(batches)
    return res


def test(model, batches, device="cpu", criterion = None): #loss_test_fold, F1_test_Fold
    model.eval()

    avg_loss = 0
        # TODO: fix for actual label count
    y_pred = np.empty((0, 19))
    y_true = np.empty((0, 19))
    
    with t.no_grad():
        for batch_i, batch in enumerate(batches):
            
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)
            
            # feed val. data through model
            outputs, probs = model(inputs)
            
            # loss
            loss = criterion(outputs, labels)
            
            # predicted labels
            predicted = t.round(probs).to(device)

            avg_loss += loss.item()
            y_pred =  np.append(y_pred, predicted.detach().to('cpu'), axis=0)
            y_true =  np.append(y_true, labels.detach().to('cpu'), axis=0)
            
    res = evaluation(y_pred, y_true)
    res["loss"] = avg_loss / len(batches)
    return res

if __name__ == "__main__":
    main()



