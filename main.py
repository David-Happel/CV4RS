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
from helper import evaluation, scalars_from_scores
import report
from arg_parser import arguments
from torchvision import transforms
import time
from torch.utils.tensorboard import SummaryWriter

#Models
from baseline_simple import C3D as bl
from CNN_LSTM_V4 import CNN_LSTM as cnn_lstm
from transformer import CNNVIT as trans
#Dataset
from dataset import DeepCropDataset, ToTensor, Normalise, labels as unique_labels, label_names

print = report.log
args = arguments()

### ====CONFIG====
device = t.device('cuda' if t.cuda.is_available() else 'cpu')
print(t.version.cuda)
print(t.cuda.is_available())

if t.cuda.is_available():
    print(f'\nUsing gpu {t.cuda.current_device()}')
else:
    print(f'\nUsing cpu')

os.chdir(os.path.dirname(os.path.abspath(__file__)))
print("Working dir: " + os.getcwd())

# Change if need to process the data
process_data = False if args.no_process_data == False else True

writer_suffix = args.name or ""
print(f'Run Name: {writer_suffix}')
writer = SummaryWriter(filename_suffix=writer_suffix, comment=writer_suffix)

#=====HANDLING ARGUMENTS=====
t_samples = args.samples or None
print(f'Samples: {t_samples}')
writer.add_text('samples', f'Samples: {t_samples}')

calculate_class_weights = False if args.no_class_weights == False else True
print(f'Class Weights: {calculate_class_weights}')
writer.add_text('Class Weights', f'Class Weights: {calculate_class_weights}')

epochs = args.epochs or 100
print(f'Epochs: {epochs}')
writer.add_text('Epochs', f'Epochs: {epochs}')

k_folds = args.folds or 5
print(f'Folds: {k_folds}')
writer.add_text('Folds', f'Folds: {k_folds}')

batch_size = args.batch_size or 10
print(f'Batch Size: {batch_size}')
writer.add_text('Batch Size', f'Batch Size: {batch_size}')

comment = args.comment or ""
print(f'Comment: {comment}')
writer.add_text('Run Comment', f'Comment: {comment}')

timepoints = args.timepoints or 6
print(f'Timepoints: {timepoints}')
writer.add_text('Timepoints', f'Timepoints: {timepoints}')

lstm_layers = args.lstm_layers or 1
print(f'lstm_layers: {lstm_layers}')
writer.add_text('LSTM_layers', f'LSTM_layers: {lstm_layers}')

trans_layers = args.trans_layers or 1
print(f'trans_layers: {trans_layers}')
writer.add_text('TRANS_layers', f'TRANS_layers: {trans_layers}')

model_name = args.model or "bl"
print(f'Model_Name: {model_name}')
model_names = ["bl", "lstm", "trans"]
models = [bl, cnn_lstm, trans]
model_class = models[model_names.index(model_name)]

bands = args.bands or "GRN,NIR,RED"
print(f'Bands: {bands}')
writer.add_text('bands', f'{bands}')
bands = bands.split(",")

#create band and times arrays
t_step = int(36 / timepoints)
times = range(0,36,t_step)

#train and test tiles
train_tiles = ["X0066_Y0041","X0067_Y0041","X0067_Y0042","X0068_Y0042","X0068_Y0043","X0069_Y0041","X0069_Y0045","X0070_Y0040","X0070_Y0045", "X0071_Y0043", "X0071_Y0045", "X0071_Y0040"]
test_tiles = ["X0071_Y0042"]

# Uncomment for testing
# train_tiles = ["X0071_Y0043"]
# test_tiles = ["X0071_Y0042"]

### Main Function
def main():
    """ Interfaces the rest of the code with the command line
    """

    #Pre-Processing
    dl = ProcessData(bands = bands, times=times)
    if process_data:
        pre_processing(dl, train_tiles=train_tiles, test_tiles=test_tiles)

    #Data Tranformations
    #mean and stdev - use helper function to re-calculate
    means = [ 736.2289, 2657.3914,  748.3498, 1886.2126, 1231.7987, 2381.1394]
    stds = [ 340017.6875, 3903396.5000,  403981.5000, 2088645.7500, 1017907.5000, 3169651.0000]
    mean_bands = ["GRN", "NIR", "RED", "SW1", "SW2", "RE3"]
    mean = []
    std = []
    for band in bands:
        index = mean_bands.index(band)
        mean.append(means[index])
        std.append(stds[index])
    mean = t.tensor(mean)
    std = t.tensor(std)
    
    data_transform = transforms.Compose([
    #Samples to tensors
     ToTensor(),
     #Normalise values for each band
     Normalise(mean, std)
    ])

    # Read in pre-processed dataset
    # data format (sample, band, time, height, width)
    print("Loading data")
    dataset = DeepCropDataset(csv_file="labels.csv", root_dir="data/prepared/train", times=times, transform=data_transform, t_samples=t_samples, bands=bands)

    # Select a random subset of the dataset to be validation data
    val_split = 0.2
    val_size = int(len(dataset) * val_split)
    train_size= len(dataset) - val_size
    train_set, val_set = t.utils.data.random_split(dataset, [train_size, val_size], generator=t.Generator().manual_seed(42))
    
    print(f'Samples: {len(dataset)} - Train: {train_size}, Val:{val_size}')

    # Calculate Class Weights
    class_weights = calc_class_weights(dataset) if calculate_class_weights else np.ones((len(d>ataset.labels)))
    print(f'Class Counts: {dataset.label_counts}')
    print(f'Class Weights: {class_weights}')

    # instantiating model 
    if model_name == 'lstm':
        model = model_class(bands=len(bands), labels=len(class_weights), time=timepoints, lstm_layers = lstm_layers).to(device)
    if model_name == 'trans':
        model = model_class(bands=len(bands), labels=len(class_weights), time=timepoints, encoder_layers=trans_layers).to(device)
    else:
        model = model_class(bands=len(bands), labels=len(class_weights), time=timepoints).to(device)
    
    # GPU parallel
    model = nn.DataParallel(model)

    #====TRAINING======
    print("===== TRAINING ======================")
    criterion = nn.BCEWithLogitsLoss(pos_weight=t.from_numpy(class_weights).to(device))
    optimizer = optim.Adam(model.parameters(), lr = 0.001)

    # Define data loaders for training and testing data in this fold
    train_batches = DataLoader(
                    train_set, 
                    batch_size=batch_size)
    val_batches = DataLoader(
                    val_set,
                    batch_size=batch_size)

    train_scores = np.empty((epochs, 45))
    val_scores =  np.empty((epochs, 45))
    score_names = None
    saved_epoch = 0 #CHECK USEFULNESS
    best_f1 = 0
    for epoch in range(epochs):  # loop over the dataset multiple times
        print(f'---- EPOCH: {epoch+1} -------------------------------')
        # train epoch
        train_score, train_score_names = train(model, train_batches, device=device, optimizer=optimizer, criterion=criterion)
        train_scores[epoch] = train_score
        
        score_names = train_score_names

        # Val epoch
        val_score, _ = predict(model, val_batches, device=device, criterion=criterion)
        val_scores[epoch] = val_score
        
        # Save the best model to date - the best global model 
        sample_f1 = val_score[list(score_names).index('samples avg_f1-score')]
        if(sample_f1 > best_f1):
            saved_epoch = epoch+1 
            save_path = f'{report.report_dir}/saved_model/model.pth'
            t.save(model.state_dict(), save_path)
            best_f1 = sample_f1
            print(f'Saved Epoch model for {epoch+1}')

    
    np.save(f'{report.report_dir}/train_scores.npy', train_scores)
    np.save(f'{report.report_dir}/val_scores.npy', val_scores)
    #helper function - transforms multidimensional scores to tensorboard
    scalars_from_scores(writer, train_scores, score_names, suffix="train")
    scalars_from_scores(writer, val_scores, score_names, suffix="val")
    
    #=====TESTING=====
    print("===== TESTING ======================")
    #Actual test performance
    # Load Testing Data
    test_dataset = DeepCropDataset(csv_file="labels.csv", root_dir="data/prepared/test", times=times, transform=data_transform, t_samples=t_samples)
    test_batches = DataLoader(
                        test_dataset,
                        batch_size=batch_size)


    #TODO load the best model

    # Test model on testing data
    test_score, _ = predict(model, test_batches, device=device, criterion=criterion)
    np.save(f'{report.report_dir}/test_score.npy', test_score)
    np.save(f'{report.report_dir}/score_names.npy', score_names)
    for score_i, score in enumerate(test_score):
        writer.add_scalar(f'{score_names[score_i]}/test', score)

    # writer.add_graph(model, test_dataset[:5].to(device))
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    writer.add_text('n_params', str(pytorch_total_params))
    pytorch_total_train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    writer.add_text('n_train_params', str(pytorch_total_train_params))

    writer.close()


def pre_processing(dl, train_tiles = ["X0071_Y0043", "X0071_Y0045", "X0071_Y0040"], test_tiles = ["X0071_Y0042"]):
    """[summary]

    Args:
        dl ([type]): [description]
        train_tiles (list, optional): [description]. Defaults to ["X0071_Y0043", "X0071_Y0045", "X0071_Y0040"].
        test_tiles (list, optional): [description]. Defaults to ["X0071_Y0042"].
    """
    print("Pre-processing data")
    #process training data 
    dl.process_tiles(train_tiles, out_dir = 'data/prepared/train/')
    #process test data 
    dl.process_tiles(test_tiles, out_dir='data/prepared/test/')

def calc_class_weights(dataset):
    """[summary]

    Args:
        dataset ([type]): [description]

    Returns:
        [type]: [description]
    """
    class_weights = len(dataset) / np.array(dataset.label_counts)
    class_weights[class_weights == float('inf')] = 1
    return class_weights

### Training Functions

def train(model, batches, device="cpu", optimizer = None, criterion = None):
    """[summary]

    Args:
        model ([type]): [description]
        batches ([type]): [description]
        device (str, optional): [description]. Defaults to "cpu".
        optimizer ([type], optional): [description]. Defaults to None.
        criterion ([type], optional): [description]. Defaults to None.

    Returns:
        [type]: [description]
    """
    model.train()

    avg_loss = 0
    y_pred = np.empty((0, len(unique_labels)))
    y_true = np.empty((0, len(unique_labels)))

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
    """[summary]

    Args:
        model ([type]): [description]
        batches ([type]): [description]
        device (str, optional): [description]. Defaults to "cpu".
        criterion ([type], optional): [description]. Defaults to None.

    Returns:
        [type]: [description]
    """

    model.eval()

    avg_loss = 0
    y_pred = np.empty((0, len(unique_labels)))
    y_true = np.empty((0, len(unique_labels)))
    
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
    start_time = time.time()
    main()
    writer.add_text('Time',str((time.time() - start_time)))
    print("--- %s seconds ---" % (time.time() - start_time))
    