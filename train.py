from enum import unique
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
from helper import reset_weights, get_labels

### CONFIG
device = t.device('cuda' if t.cuda.is_available() else 'cpu')

if t.cuda.is_available():
    print(f'\nUsing gpu {t.cuda.current_device()}')
else:
    print(f'\nUsing cpu')

# Change if need to process the data
process_data = False

#create band and times arrays
t_start = 1
t_stop = 37
t_step = 6
times = range(t_start,t_stop,t_step)
bands = ["GRN", "NIR", "RED"]
labels, label_names = get_labels()

t_samples = 10

# Change if need to re-process the data
process_data = False

### Main Function
def main():
    #PREPARE DATA 
    dl = ProcessData(bands = bands, times=times)
    if process_data:
        #process training data 
        dl.process_tile("X0071_Y0043", out_dir = 'data/prepared/train/')
        #process test data 
        dl.process_tile("X0071_Y0043", out_dir='data/prepared/test/')

    #create dataset
    #data format (sample, band, time, height, width)

    train_data, train_labels = dl.read_dataset(out_dir='data/prepared/train/', t_samples=t_samples)
    #converting to tensor datasets
    train_ds = TensorDataset(train_data , train_labels)

    labels_n = train_labels.shape[1]
    print("Labels: ", labels_n)


    #TRAINING
    criterion = nn.BCEWithLogitsLoss()

    epochs = 2
    k_folds = 5

    kfold = KFold(n_splits=k_folds, shuffle=True)

    train_scores = np.array([])
    val_scores =  np.array([])
    for fold, (train_ids, test_ids) in enumerate(kfold.split(train_ds)):

        print("fold:", fold)
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


        for epoch in range(epochs):  # loop over the dataset multiple times
            # TRAIN EPOCH
            train_score = train(model, train_batches, device=device, optimizer=optimizer, criterion=criterion)
            train_scores = np.append(train_scores, [train_score])
            print('Train Epoch: {}/{} \tLoss: {:.8f} \tAccuracy: {}/{} ({:.4f}%), F1: {:.4f} \n'.format(
                    epoch + 1, epochs,  # epoch / epochs
                    train_score["loss"], # loss for that epoch
                    train_score["correct"], labels_n * len(train_ids),
                    train_score["accuracy"],
                    train_score["f1"],                                                      
                    end='\r'))
            
            # TEST EPOCH
            test_score = test(model, test_batches, device=device, criterion=criterion)
            val_scores = np.append(val_scores, [test_score])
            print('Validation Epoch: {}/{} \tLoss: {:.8f} \tAccuracy: {}/{} ({:.4f}%), F1: {:.4f} \n'.format(
                    epoch + 1, epochs,  # epoch / epochs
                    test_score["loss"], # loss for that epoch
                    test_score["correct"], labels_n * len(test_ids),
                    test_score["accuracy"],
                    test_score["f1"],                                                        
                    end='\r'))
            #TODO: Check if epoch is better than last and save model if so.
            
        # Process is complete.
        print('Training process has finished. Saving trained model.')

        # Saving the model
        save_path = f'./models/saved/model-fold-{fold}.pth'
        t.save(model.state_dict(), save_path)


    # Print fold results
    print(f'K-FOLD CROSS VALIDATION RESULTS FOR {k_folds} FOLDS')
    loss_sum = sum(map(lambda fold: fold["loss"],val_scores))
    print(f'Average Loss: {loss_sum/len(val_scores)}')

    acc_sum = sum(map(lambda fold: fold["accuracy"],val_scores))
    print(f'Average Accuracy: {acc_sum/len(val_scores)}')

    f1_sum = sum(map(lambda fold: fold["f1"],val_scores))
    print(f'Average f1: {f1_sum/len(val_scores)}')

    #TEST EVALUATION
    ## #TODO  work on test set

    # test_data, test_labels = dl.read_dataset(out_dir='data/prepared/test/')
    # test_ds = TensorDataset(test_data , test_labels)



### Training Functions

# train the model
def train(model, batches, device="cpu", optimizer = None, criterion = None):
    model.train()
    
    # for correct labels
    correct = 0 
    samples_n = 0

    for batch_i, batch in enumerate(batches):
        print('Batch: {}/{}'.format(batch_i+1, len(batches)))
        
        # batch is a list of 10 [inputs, labels]  -> e.g. len(labels) = 10
        inputs, labels = batch
        inputs, labels = inputs.to(device), labels.to(device)
        samples_n += inputs.shape[0]
        
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
        
        # corretly predicted labels
        correct += (predicted == labels).sum().item()
        #print(correct)
    
    
    #accuracy
    accuracy = 100 * correct / (len(labels[0]) * samples_n)
    
    # F1 score for the batch
    # f1 = f1_score(labels.detach().to('cpu'), predicted.detach().to('cpu'), average = None)
    f1=0
    #TODO: Replace with standardised function to compute scores
    return {"accuracy":accuracy, "loss": loss.item(), "f1":f1, "correct":correct}


def test(model, batches, device="cpu", criterion = None): #loss_test_fold, F1_test_Fold
    model.eval()

    correct = 0
    samples_n = 0
    
    with t.no_grad():
        for batch_i, batch in enumerate(batches):
            
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)
            samples_n += inputs.shape[0]
            
            # feed val. data through model
            outputs, probs = model(inputs)
            
            # loss
            loss = criterion(outputs, labels)
            
            # predicted labels
            predicted = t.round(probs).to(device)

            # number of correctly predicted image labels
            correct += (predicted == labels).sum().item()
                
    #accuracy
    accuracy = 100 * correct / (len(labels[0]) * samples_n)

    # F1 score for the batch
    # f1 = f1_score(labels.detach().to('cpu'), predicted.detach().to('cpu'), average = None)
    f1=0
    #TODO: Replace with standardised function to compute scores
    return {"accuracy":accuracy, "loss": loss.item(), "f1":f1, "correct":correct}

if __name__ == "__main__":
    main()



