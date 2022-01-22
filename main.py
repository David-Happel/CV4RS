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
import flatten_json
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, accuracy_score
from processdata import ProcessData
from helper import reset_weights, get_labels, evaluation, scalars_from_scores
import report
from arg_parser import arguments

from torch.utils.tensorboard import SummaryWriter
#Models
from baseline_simple import C3D as bl
from CNN_LSTM_V1 import CNN_LSTM as cnn_lstm
from transformer import CNNVIT as trans

from dataset import DeepCropDataset, ToTensor

print = report.log

args = arguments()

### CONFIG
device = t.device('cuda' if t.cuda.is_available() else 'cpu')

if t.cuda.is_available():
    print(f'\nUsing gpu {t.cuda.current_device()}')
else:
    print(f'\nUsing cpu')

os.chdir(os.path.dirname(os.path.abspath(__file__)))
print("Working dir: " + os.getcwd())

# Change if need to process the data
process_data = False if args.no_process_data == False else True

#Restriction of samples to take
t_samples = args.samples or None
print(f'Samples: {t_samples}')

calculate_class_weights = False if args.no_class_weights == False else True
print(f'Class Weights: {calculate_class_weights}')

epochs = args.epochs or 100
print(f'Epochs: {epochs}')

k_folds = args.folds or 5
print(f'Folds: {k_folds}')

batch_size = args.batch_size or 10
print(f'Batch Size: {batch_size}')

writer_suffix = args.name or ""
print(f'Run Comment: {writer_suffix}')

timepoints = args.timepoints or 6
print(f'Timepoints: {timepoints}')

model_name = args.model or "bl"
print(f'Model_Name: {model_name}')
model_names = ["bl", "lstm", "trans"]
models = [bl, cnn_lstm, trans]
model_class = models[model_names.index(model_name)]

#create band and times arrays
t_step = int(36 / timepoints)
times = range(0,36,t_step)
bands = ["GRN", "NIR", "RED"]
labels, label_names = get_labels()

#train and test tiles
# train_tiles = ["X0066_Y0041","X0067_Y0041","X0067_Y0042","X0068_Y0043","X0069_Y0041","X0069_Y0042","X0069_Y0045","X0070_Y0040","X0070_Y0045", "X0071_Y0043", "X0071_Y0045", "X0071_Y0040"]
# test_tiles = ["X0071_Y0042"]

# Uncomment for testing
# train_tiles = ["X0071_Y0043"]
# test_tiles = ["X0071_Y0042"]

writer = SummaryWriter(filename_suffix=writer_suffix, comment=writer_suffix)

### Main Function
def main():

    #PRE - Processing
    dl = ProcessData(bands = bands, times=times)
    if process_data:
        pre_processing(dl, train_tiles=train_tiles, test_tiles=test_tiles)

    # Read in pre-processed dataset
    # data format (sample, band, time, height, width)
    print("Loading data")
    train_dataset = DeepCropDataset(csv_file="labels.csv", root_dir="data/prepared/train", times=times, transform=ToTensor(), t_samples=t_samples)
    print(f'Samples: {len(train_dataset)}')

    # Calculate Class Weights
    class_weights, class_counts = calc_class_weights(train_labels) if calculate_class_weights else t.from_numpy(np.ones((len(train_dataset.labels)))),[]
    print(f'Class Counts: {class_counts}')
    print(f'Class Weights: {class_weights}')

    #TRAINING
    criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights.to(device))

    kfold = KFold(n_splits=k_folds, shuffle=True)

    train_scores = np.empty((k_folds, epochs, 101))
    val_scores =  np.empty((k_folds, epochs, 101))
    score_names = None
    for fold, (train_ids, test_ids) in enumerate(kfold.split(train_dataset)):

        print(f'======= FOLD: {fold+1} ==================================')
        # Sample elements randomly from a given list of ids, no replacement.
        train_subsampler = SubsetRandomSampler(train_ids)
        test_subsampler = SubsetRandomSampler(test_ids)

        # Define data loaders for training and testing data in this fold
        train_batches = DataLoader(
                        train_dataset, 
                        batch_size=batch_size, sampler=train_subsampler)
        test_batches = DataLoader(
                        train_dataset,
                        batch_size=batch_size, sampler=test_subsampler)

        #model selection
        model = model_class(bands=len(bands), labels=len(class_weights), device=device).to(device)

        optimizer = optim.Adam(model.parameters(), lr = 0.001)

        saved_epoch = 0
        best_f1 = 0
        for epoch in range(epochs):  # loop over the dataset multiple times
            print(f'---- EPOCH: {epoch+1} -------------------------------')
            # TRAIN EPOCH
            train_score, train_score_names = train(model, train_batches, device=device, optimizer=optimizer, criterion=criterion)
            train_scores[fold, epoch] = train_score
            
            score_names = train_score_names

            # TEST EPOCH
            test_score, _ = predict(model, test_batches, device=device, criterion=criterion)
            val_scores[fold, epoch] = test_score
            
            # Save model if best f1 score of epoch
            sample_f1 = test_score[list(score_names).index('samples avg_f1-score')]
            if(sample_f1 > best_f1):
                saved_epoch = epoch+1
                save_path = f'{report.report_dir}/saved_model/model-fold-{fold+1}.pth'
                t.save(model.state_dict(), save_path)
                best_f1 = sample_f1
                print(f'Saved Epoch model for {epoch+1}')

        print(f'Model saved for Fold {fold+1}: epoch {saved_epoch}')
    
    np.save(f'{report.report_dir}/train_scores.npy', train_scores)
    np.save(f'{report.report_dir}/val_scores.npy', val_scores)
    #helper function - transforms multidimensional scores to tensorboard
    scalars_from_scores(writer, train_scores, score_names, suffix="train")
    scalars_from_scores(writer, val_scores, score_names, suffix="val")
    

    print("===== TESTING ======================")
    #Actual test performance
    # Load Testing Data
    test_dataset = DeepCropDataset(csv_file="labels.csv", root_dir="data/prepared/test", times=times, transform=ToTensor(), t_samples=t_samples)
    test_batches = DataLoader(
                        test_dataset,
                        batch_size=batch_size)

    # Test latest fold model on testing data
    test_score, _ = predict(model, test_batches, device=device, criterion=criterion)
    np.save(f'{report.report_dir}/test_score.npy', test_score)
    np.save(f'{report.report_dir}/score_names.npy', score_names)

    writer.add_graph(model, test_dataset[:5].to(device))
    writer.close()


def pre_processing(dl, train_tiles = ["X0071_Y0043", "X0071_Y0045", "X0071_Y0040"], test_tiles = ["X0071_Y0042"]):
    print("Pre-processing data")
    #process training data 
    dl.process_tiles(train_tiles, out_dir = 'data/prepared/train/')
    #process test data 
    dl.process_tiles(test_tiles, out_dir='data/prepared/test/')

def calc_class_weights(labels):
    class_counts = t.sum(labels, axis=0)
    class_weights = labels.shape[0] / class_counts
    class_weights[class_weights == float('inf')] = 1
    return class_weights, class_counts

### Training Functions

def train(model, batches, device="cpu", optimizer = None, criterion = None):
    model.train()

    avg_loss = 0
    y_pred = np.empty((0, len(get_labels()[0])))
    y_true = np.empty((0, len(get_labels()[0])))

    for batch_i, batch in enumerate(batches):
        print('Batch: {}/{}'.format(batch_i+1, len(batches)))
        
        # batch is a list of [inputs, labels]
        inputs, labels = batch
        inputs, labels = inputs.to(device), labels.to(device)
        
        # improve params. for the batch
        optimizer.zero_grad()
        
        # forward + backward + optimize  for the batch
        # outputs =  logits, probs ) probabilities from sigmoid layer
        outputs, probs = model(inputs)
        
        # compute loss using logits
        loss = criterion(outputs, labels)
            
        # do backprop. for the batch
        loss.backward()

        # update step for the batch
        optimizer.step()
            
        # predicted labels for F1 score
        predicted = t.round(probs).to(device)

        avg_loss += loss.item()

        y_pred =  np.append(y_pred, predicted.detach().to('cpu'), axis=0)
        y_true =  np.append(y_true, labels.detach().to('cpu'), axis=0)
        
    scores = dict({"loss": avg_loss / len(batches)})
    return evaluation(y_pred, y_true, initial=scores)

#Make predictions
def predict(model, batches, device="cpu", criterion = None): #loss_test_fold, F1_test_Fold
    model.eval()

    avg_loss = 0
    y_pred = np.empty((0, len(get_labels()[0])))
    y_true = np.empty((0, len(get_labels()[0])))
    
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
            
    scores = dict({"loss": avg_loss / len(batches)})
    return evaluation(y_pred, y_true, initial=scores)

if __name__ == "__main__":
    main()