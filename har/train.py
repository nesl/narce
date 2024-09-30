from tqdm import tqdm
import copy
import numpy as np
import torch
from torch import nn

from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay, accuracy_score, balanced_accuracy_score

import matplotlib.pyplot as plt

from nar_model import InferenceParams

import wandb

def cross_validation_folds(dataset, k_folds):
    """
        Args:
            dataset: A Dataset class of training data to split into train & validation set
            k_folds: An integer representing the number of cross validation folds
        Returns:
            folds: A list of tuples representing (train, validation) indices
        """
    kf = KFold(n_splits=k_folds, shuffle=True)
    folds = []
    for train_idx, valid_idx in kf.split(dataset):
        folds.append((train_idx, valid_idx))
    return folds


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        print('Early-stop counter:', self.counter)
        return False


def train(model, train_loader, val_loader, n_epochs, lr, criterion, save_path, min_delta=1e-4, src_mask=None, multi_task=False, criterion2=nn.CrossEntropyLoss(), device='cpu'):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=[0.9, 0.95], weight_decay=0.1, )
    # early_stopper = EarlyStopper(patience=50, min_delta=0.001)
    early_stopper = EarlyStopper(patience=30, min_delta=min_delta)
    summary = {'train_loss': [[] for _ in range(n_epochs)], 
               'train_label_acc': [[] for _ in range(n_epochs)], 
               'train_state_acc': [[] for _ in range(n_epochs)], 
               'val_loss': [[] for _ in range(n_epochs)], 
               'val_label_acc': [[] for _ in range(n_epochs)],
               'val_state_acc': [[] for _ in range(n_epochs)]}

    for e in tqdm(range(n_epochs)):
        print("Epoch:", e)
        model.to(device)
        model.train()
        for i, batch in enumerate(train_loader):
            optimizer.zero_grad()
            data = batch[0].to(device)
            labels = batch[1].to(device)
            if multi_task:
                in_states = batch[2].to(device)
                out_states = batch[3].to(device)

            # Run the Net
            if src_mask is not None: # For transformer
                src_mask = src_mask.to(device)
                if multi_task:
                    x,s = model(data, in_states, src_mask)
                else:
                    x = model(data, src_mask)
            else:
                if multi_task:
                    x,s = model(data, in_states)
                else:
                    x = model(data)

            x = x.transpose(-1,1) # sequence cross entropy loss accepts input of dimension (N, C, L)
            if multi_task:
                s = s.transpose(-1,1)

            # Calculate loss
            criterion = criterion.to(device)
            loss = criterion(x, labels)
            if multi_task:
                state_loss = criterion2(s, out_states)
                loss = 0.5 * loss + 0.5 * state_loss

            # Optimize net
            loss.backward()
            optimizer.step()
            summary['train_loss'][e].append(loss.item())

            # Calculate accuracy
            _, xpred = x.data.topk(1, dim=1)
            xpred = xpred.squeeze(1)
            acc = torch.sum(xpred == labels)/(x.shape[0] * x.shape[-1])
            summary['train_label_acc'][e].append(acc.item())
            if multi_task:
                _, spred = s.data.topk(1, dim=1)
                spred = spred.squeeze(1)
                acc = torch.sum(spred == out_states)/(s.shape[0] * s.shape[-1])
                summary['train_state_acc'][e].append(acc.item())

        # Test on validation set
        # model.to('cpu') # mamba has some issue on cpu
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                # data = batch[0]
                # labels = batch[1]
                # if multi_task:
                #     in_states = batch[2]
                #     out_states = batch[3]
                data = batch[0].to(device)
                labels = batch[1].to(device)
                if multi_task:
                    in_states = batch[2].to(device)
                    out_states = batch[3].to(device)

                if src_mask is not None: # For transformer
                    # src_mask = src_mask.to('cpu')
                    if multi_task:
                        x,s = model(data, in_states, src_mask)
                    else:
                        x = model(data, src_mask)
                else:
                    if multi_task:
                        x,s = model(data, in_states)
                    else:
                        x = model(data)

                x = x.transpose(-1,1)
                if multi_task:
                    s = s.transpose(-1,1)
                
                # Calculate validation loss and accuracy
                # criterion = criterion.to('cpu')
                loss = criterion(x, labels)
                if multi_task:
                    state_loss = criterion2(s, out_states)
                    loss = 0.5 * loss + 0.5 * state_loss

                summary['val_loss'][e].append(loss.item())

                _, xpred = x.data.topk(1, dim=1)
                xpred = xpred.squeeze(1)
                xacc = torch.sum(xpred == labels)/(x.shape[0] * x.shape[-1])
                summary['val_label_acc'][e].append(xacc.item())
                if multi_task:
                    _, spred = s.data.topk(1, dim=1)
                    spred = spred.squeeze(1)
                    sacc = torch.sum(spred == out_states)/(s.shape[0] * s.shape[-1])
                    summary['val_state_acc'][e].append(sacc.item())
            
        print('pred labels',xpred)
        if not multi_task:
            print('Training Loss: {}, Training Accuracy - Label: {}, Validation Loss: {}, Validation Accuracy - Label: {}'.format(
                np.mean(summary['train_loss'][e]), 
                np.mean(summary['train_label_acc'][e]), 
                np.mean(summary['val_loss'][e]), 
                np.mean(summary['val_label_acc'][e])))
        else:
            print('pred states', spred)
            print('Training Loss: {}, Training Accuracy - Label: {}, Training Accuracy - State: {}, \nValidation Loss: {}, Validation Accuracy - Label: {}, Validation Accuracy - State: {}'.format(
                np.mean(summary['train_loss'][e]), 
                np.mean(summary['train_label_acc'][e]), 
                np.mean(summary['train_state_acc'][e]), 
                np.mean(summary['val_loss'][e]), 
                np.mean(summary['val_label_acc'][e]),
                np.mean(summary['val_state_acc'][e]), ))

        if np.mean(summary['val_loss'][e]) < early_stopper.min_validation_loss:
            best_model = copy.deepcopy(model)

        # Save the best model every 1000 epochs
        if e % 1000 == 0 and e > 0:
            torch.save(best_model.state_dict(), save_path)

        if early_stopper.early_stop(np.mean(summary['val_loss'][e])):  
            print("Early-stop at epoch:", e)           
            break

    
    # Update the model with the best one after training
    model = best_model
    torch.save(model.state_dict(), save_path)

    return summary

@torch.inference_mode()
def test(model, data_loader, criterion, save_fig_dir, src_mask=None, multi_task=False, criterion2=nn.CrossEntropyLoss(), device='cpu'):
    model.eval()
    model.to(device)
    summary = {'loss': [] , 'acc_labels': [], 'acc_states': []}

    all_labels_pred = []
    all_labels = []
    all_states_pred = []
    all_states = []
    all_simple_labels_pred = []
    all_simple_labels = []

    for i, batch in enumerate(tqdm(data_loader)):
        # data = batch[0]
        # labels = batch[1]
        # if multi_task:
        #     in_states = batch[2]
        #     out_states = batch[3]
        data = batch[0].to(device)
        labels = batch[1]
        if multi_task:
            in_states = batch[2].to(device)
            out_states = batch[3]

        # Run the Net
        if src_mask is not None: # For transformer
            src_mask = src_mask.to(device)
            if multi_task:
                x,s = model(data, in_states, src_mask)
            else:
                x = model(data, src_mask)
        else:
            if multi_task:
                x,s = model(data, in_states)
            else:
                x = model(data)
        
        x = x.transpose(-1,1)
        if multi_task:
            s = s.transpose(-1,1)

        # Calculate validation loss and accuracy
        criterion = criterion.to('cpu')
        x = x.cpu()
        loss = criterion(x, labels) # We can add iterative training: use ignore_index=ignore_index in los function
        if multi_task:
            s = s.cpu()
            state_loss = criterion2(s, out_states)
            loss = 0.5 * loss + 0.5 * state_loss
        summary['loss'].append(loss.item())

        _, xpred = x.data.topk(1, dim=1)
        xpred = xpred.squeeze(1)
        xacc = torch.sum(xpred == labels)/(x.shape[0] * x.shape[-1])
        summary['acc_labels'].append(xacc.item())
        if multi_task:
            _, spred = s.data.topk(1, dim=1)
            spred = spred.squeeze(1)
            sacc = torch.sum(spred == out_states)/(s.shape[0] * s.shape[-1])
            summary['acc_states'].append(sacc.item())

        # Visualize the result
        for j in range(len(xpred)):
            print("Pred label:",xpred[j])
            print("True label:", labels[j])
            if multi_task:
                print("Pred state:",spred[j])
                print("True state:", out_states[j])

        all_labels_pred.append(xpred.reshape(-1))
        all_labels.append(labels.reshape(-1))
        if multi_task:
            all_states_pred.append(spred.reshape(-1))
            all_states.append(out_states.reshape(-1))

        simple_labels_pred = xpred.max(dim=1).values
        simple_labels = labels.max(dim=1).values
        all_simple_labels_pred.append(simple_labels_pred)
        all_simple_labels.append(simple_labels)

    all_labels_pred = np.concatenate(all_labels_pred)
    all_labels = np.concatenate(all_labels)
    all_simple_labels_pred = np.concatenate(all_simple_labels_pred)
    all_simple_labels = np.concatenate(all_simple_labels)
    print(all_simple_labels_pred[0],all_simple_labels[0])

    # Timewise evaluation

    f1 = f1_score(all_labels, all_labels_pred, average=None)
    f1_all = f1_score(all_labels, all_labels_pred, average='macro')
    f1_pos = f1_score(all_labels, all_labels_pred, labels=[1,2,3], average='macro')

    precision = precision_score(all_labels, all_labels_pred, average=None)
    recall = recall_score(all_labels, all_labels_pred, average=None)
    precision_avg = precision_score(all_labels, all_labels_pred, average='macro')
    recall_avg = recall_score(all_labels, all_labels_pred, average='macro')

    print('Total loss: {}'.format(np.mean(summary['loss'])))
    print('CE labels - Accuracy: {}, F1: {}, F1_all: {}, F1_positive: {}, Precision: {}, Avg_P: {}, Recall: {}, Avg_R: {}'.format(
                                                                                                                np.mean(summary['acc_labels']),
                                                                                                                f1,
                                                                                                                f1_all,
                                                                                                                f1_pos,
                                                                                                                precision,
                                                                                                                precision_avg,
                                                                                                                recall,
                                                                                                                recall_avg
                                                                                                                ))
    if multi_task:
        all_states_pred = np.concatenate(all_states_pred)
        all_states = np.concatenate(all_states)
        f1 = f1_score(all_states, all_states_pred, average='macro')

        precision = precision_score(all_states, all_states_pred, average=None)
        recall = recall_score(all_states, all_states_pred, average=None)
        precision_avg = precision_score(all_states, all_states_pred, average='macro')
        recall_avg = recall_score(all_states, all_states_pred, average='macro')
        print('FSM states - Accuracy: {}, F1: {}, Precision: {}, Avg_P: {}, Recall: {}, Avg_R: {}'.format(
                                                                                                            np.mean(summary['acc_states']),
                                                                                                            f1,
                                                                                                            precision,
                                                                                                            precision_avg,
                                                                                                            recall,
                                                                                                            recall_avg
                                                                                                            ))
        
    # Sample-wise evaluation
    simple_f1_all = f1_score(y_true=all_simple_labels, y_pred=all_simple_labels_pred, average='macro')
    acc_score = accuracy_score(y_true=all_simple_labels, y_pred=all_simple_labels_pred)
    balanced_acc_score = balanced_accuracy_score(y_true=all_simple_labels, y_pred=all_simple_labels_pred)
    cm = confusion_matrix(y_true=all_simple_labels, y_pred=all_simple_labels_pred)
    cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No_Event", "CE_1", "CE_2", "CE_3"])
    cm_display.plot(cmap=plt.cm.Blues)
    plt.savefig(save_fig_dir) 

    print('CE labels sample level - Accuracy: {}, Balanced_Accuracy: {}, F1_all: {}'.format(acc_score,
                                                                               balanced_acc_score,
                                                                               simple_f1_all
                                                                               ))

        

def test_iterative(model, data_loader, criterion, src_mask=None, criterion2=nn.CrossEntropyLoss()):
    ''' The model's no longer provided with ground-truth current state s_t at inference time - 
        Instead, it needs to predict current state s'_t at each time t by itself, and use  s'_t to predict CE label y_t and the next state s'_t+1 iteratively.'''
    model.eval()
    model.to('cpu')
    summary = {'loss': [] , 'acc_labels': [], 'acc_states': []}

    all_labels_pred = []
    all_labels = []
    all_states_pred = []
    all_states = []

    T = data_loader.dataset.data.shape[1]

    for _, batch in enumerate(tqdm(data_loader)):
        in_states = 0
        for i in range(T):
            data = batch[0]
            labels = batch[1]
            out_states = batch[3]
            
            if i == 0:
                in_states = batch[2]
            # otherwise, use states predicted in the last iteration

            # Use seqeuntial batch mask to get state prediction one by one
            src_padding_mask = torch.ones([len(data), T], dtype=torch.bool, device='cpu')
            src_padding_mask[:, :i+1] = False
            
            # Run the Net
            with torch.no_grad():
                if src_mask is not None: # For transformer
                    src_mask = src_mask.to('cpu')
                    x,s = model(data, in_states, src_mask, src_padding_mask)
                else:
                    x,s = model(data, in_states)


            # Update the next input state by shifting the predicted output state
            _, spred = s.data.topk(1, dim=-1)
            spred = spred.squeeze(-1)
            in_states[:, i+1:] = spred[:, i:-1]

            # Calculate validation loss and accuracy after the target sequence prediction is complete
            if i == T - 1:
                x = x.transpose(-1,1)
                s = s.transpose(-1,1)

                criterion = criterion.to('cpu')
                loss = criterion(x, labels)
                state_loss = criterion2(s, out_states)
                loss = 0.5 * loss + 0.5 * state_loss
                summary['loss'].append(loss.item())

                _, xpred = x.data.topk(1, dim=1)
                xpred = xpred.squeeze(1)
                xacc = torch.sum(xpred == labels)/(x.shape[0] * x.shape[-1])
                summary['acc_labels'].append(xacc.item())
                
                _, spred = s.data.topk(1, dim=1)
                spred = spred.squeeze(1)
                sacc = torch.sum(spred == out_states)/(s.shape[0] * s.shape[-1])
                summary['acc_states'].append(sacc.item())

                # Visualize the result
                for j in range(len(xpred)):
                    print("Pred label:",xpred[j])
                    print("True label:", labels[j])
                    print("Pred state:",spred[j])
                    print("True state:", out_states[j])

                all_labels_pred.append(xpred.reshape(-1))
                all_labels.append(labels.reshape(-1))
                all_states_pred.append(spred.reshape(-1))
                all_states.append(out_states.reshape(-1))

    all_labels_pred = np.concatenate(all_labels_pred)
    all_labels = np.concatenate(all_labels)

    f1_all = f1_score(all_labels, all_labels_pred, average='macro')
    f1_pos = f1_score(all_labels, all_labels_pred, labels=[1,2,3], average='macro')

    precision = precision_score(all_labels, all_labels_pred, average=None)
    recall = recall_score(all_labels, all_labels_pred, average=None)
    precision_avg = precision_score(all_labels, all_labels_pred, average='macro')
    recall_avg = recall_score(all_labels, all_labels_pred, average='macro')

    print('Total loss: {}'.format(np.mean(summary['loss'])))
    print('CE labels - Accuracy: {}, F1_all: {}, F1_positive: {}, Precision: {}, Avg_P: {}, Recall: {}, Avg_R: {}'.format(
                                                                                                                np.mean(summary['acc_labels']),
                                                                                                                f1_all,
                                                                                                                f1_pos,
                                                                                                                precision,
                                                                                                                precision_avg,
                                                                                                                recall,
                                                                                                                recall_avg
                                                                                                                ))
    all_states_pred = np.concatenate(all_states_pred)
    all_states = np.concatenate(all_states)
    f1 = f1_score(all_states, all_states_pred, average='macro')

    precision = precision_score(all_states, all_states_pred, average=None)
    recall = recall_score(all_states, all_states_pred, average=None)
    precision_avg = precision_score(all_states, all_states_pred, average='macro')
    recall_avg = recall_score(all_states, all_states_pred, average='macro')
    print('FSM states - Accuracy: {}, F1: {}, Precision: {}, Avg_P: {}, Recall: {}, Avg_R: {}'.format(
                                                                                                        np.mean(summary['acc_states']),
                                                                                                        f1,
                                                                                                        precision,
                                                                                                        precision_avg,
                                                                                                        recall,
                                                                                                        recall_avg
                                                                                                        ))
    

def train_narce(model, 
                train_loader, 
                val_loader, 
                n_epochs, 
                lr, 
                criterion, 
                save_path, 
                is_train_adapter=False, 
                has_state=False, 
                state_criterion=nn.CrossEntropyLoss(), 
                device='cpu'):
    
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    # early_stopper = EarlyStopper(patience=50, min_delta=0.001)
    if is_train_adapter:
        early_stopper = EarlyStopper(patience=30, min_delta=1e-4)
    else:
        early_stopper = EarlyStopper(patience=30, min_delta=1e-6)
    summary = {'train_loss': [[] for _ in range(n_epochs)], 
               'train_label_acc': [[] for _ in range(n_epochs)], 
               'train_state_acc': [[] for _ in range(n_epochs)], 
               'val_loss': [[] for _ in range(n_epochs)], 
               'val_label_acc': [[] for _ in range(n_epochs)],
               'val_state_acc': [[] for _ in range(n_epochs)]}


    for e in tqdm(range(n_epochs)):
        print("Epoch:", e)
        model.to(device)
        model.train()
        for i, batch in enumerate(train_loader):
            optimizer.zero_grad()
            data = batch[0].to(device)
            labels = batch[1].to(device)
            if has_state:
                in_states = batch[2].to(device)
                out_states = batch[3].to(device)

            # Run the Net
            if has_state:
                x, s = model(data, in_states)
            else:
                x = model(data)

            x = x.transpose(-1,1) # sequence cross entropy loss accepts input of dimension (N, C, L)
            if has_state:
                s = s.transpose(-1,1)


            # Calculate loss
            criterion = criterion.to(device)
            loss = criterion(x, labels)
            if has_state:
                state_loss = state_criterion(s, out_states)
                loss = 0.5 * loss + 0.5 * state_loss

            # Optimize net
            loss.backward()
            optimizer.step()
            summary['train_loss'][e].append(loss.item())

            # Calculate accuracy
            _, xpred = x.data.topk(1, dim=1)
            xpred = xpred.squeeze(1)
            acc = torch.sum(xpred == labels)/(x.shape[0] * x.shape[-1])
            summary['train_label_acc'][e].append(acc.item())
            if has_state:
                _, spred = s.data.topk(1, dim=1)
                spred = spred.squeeze(1)
                acc = torch.sum(spred == out_states)/(s.shape[0] * s.shape[-1])
                summary['train_state_acc'][e].append(acc.item())


        # Test on validation set
        # model.to('cpu') # mamba has some issue on cpu
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                # data = batch[0]
                # labels = batch[1]
                # if multi_task:
                #     in_states = batch[2]
                #     out_states = batch[3]
                data = batch[0].to(device)
                labels = batch[1].to(device)
                if has_state:
                    in_states = batch[2].to(device)
                    out_states = batch[3].to(device)

                if has_state:
                    x, s = model(data, in_states)
                else:
                    x = model(data)

                x = x.transpose(-1,1)
                if has_state:
                    s = s.transpose(-1,1)
                
                # Calculate validation loss and accuracy
                # criterion = criterion.to('cpu')
                loss = criterion(x, labels)
                if has_state:
                    state_loss = state_criterion(s, out_states)
                    loss = 0.5 * loss + 0.5 * state_loss

                summary['val_loss'][e].append(loss.item())

                _, xpred = x.data.topk(1, dim=1)
                xpred = xpred.squeeze(1)
                xacc = torch.sum(xpred == labels)/(x.shape[0] * x.shape[-1])
                summary['val_label_acc'][e].append(xacc.item())
                if has_state:
                    _, spred = s.data.topk(1, dim=1)
                    spred = spred.squeeze(1)
                    acc = torch.sum(spred == out_states)/(s.shape[0] * s.shape[-1])
                    summary['val_state_acc'][e].append(acc.item())
            
        print('pred labels',xpred)
        if not has_state:
            print('Training Loss: {}, Training Accuracy - Label: {}, Validation Loss: {}, Validation Accuracy - Label: {}'.format(
                np.mean(summary['train_loss'][e]), 
                np.mean(summary['train_label_acc'][e]), 
                np.mean(summary['val_loss'][e]), 
                np.mean(summary['val_label_acc'][e])))
        else:
            print('pred states', spred)
            print('Training Loss: {}, Training Accuracy - Label: {}, Training Accuracy - State: {}, \nValidation Loss: {}, Validation Accuracy - Label: {}, Validation Accuracy - State: {}'.format(
                np.mean(summary['train_loss'][e]), 
                np.mean(summary['train_label_acc'][e]), 
                np.mean(summary['train_state_acc'][e]), 
                np.mean(summary['val_loss'][e]), 
                np.mean(summary['val_label_acc'][e]),
                np.mean(summary['val_state_acc'][e]), ))
            
        # Log to wandb for plots    
        metrics = {"train/train_loss": np.mean(summary['train_loss'][e]),
                   "train/train_acc": np.mean(summary['train_label_acc'][e]),
                   "train/epoch": e,
                   "val/val_loss": np.mean(summary['val_loss'][e]),
                   "val/val_acc": np.mean(summary['val_label_acc'][e]),
                   "val/epoch": e,
                   }
        wandb.log(metrics)
        
        if np.mean(summary['val_loss'][e]) < early_stopper.min_validation_loss:
            best_model = copy.deepcopy(model)

        # Save the best model every 1000 epochs
        if e % 1000 == 0 and e > 0:
            torch.save(best_model.state_dict(), save_path)

        if early_stopper.early_stop(np.mean(summary['val_loss'][e])):  
            print("Early-stop at epoch:", e)           
            break

    
    # Save model after training
    torch.save(best_model.state_dict(), save_path)

    return summary


def test_narce(model, data_loader, criterion, device='cpu'):
    model.eval()
    model.to(device)
    summary = {'loss': [] , 'acc_labels': []}

    all_labels_pred = []
    all_labels = []

    for i, batch in enumerate(tqdm(data_loader)):
        data = batch[0].to(device)
        labels = batch[1]

        # Run the Net
        with torch.no_grad():
            x = model(data)
            x = x.transpose(-1,1)

        # Calculate validation loss and accuracy
        criterion = criterion.to('cpu')
        x = x.cpu()
        loss = criterion(x, labels) # We can add iterative training: use ignore_index=ignore_index in los function
        summary['loss'].append(loss.item())

        _, xpred = x.data.topk(1, dim=1)
        xpred = xpred.squeeze(1)
        xacc = torch.sum(xpred == labels)/(x.shape[0] * x.shape[-1])
        summary['acc_labels'].append(xacc.item())

        # Visualize the result
        for j in range(len(xpred)):
            print("Pred label:",xpred[j])
            print("True label:", labels[j])

        all_labels_pred.append(xpred.reshape(-1))
        all_labels.append(labels.reshape(-1))

    all_labels_pred = np.concatenate(all_labels_pred)
    all_labels = np.concatenate(all_labels)

    f1_all = f1_score(all_labels, all_labels_pred, average='macro')
    f1_pos = f1_score(all_labels, all_labels_pred, labels=[1,2,3], average='macro')

    precision = precision_score(all_labels, all_labels_pred, average=None)
    recall = recall_score(all_labels, all_labels_pred, average=None)
    precision_avg = precision_score(all_labels, all_labels_pred, average='macro')
    recall_avg = recall_score(all_labels, all_labels_pred, average='macro')

    print('Total loss: {}'.format(np.mean(summary['loss'])))
    print('CE labels - Accuracy: {}, F1_all: {}, F1_positive: {}, Precision: {}, Avg_P: {}, Recall: {}, Avg_R: {}'.format(
                                                                                                                np.mean(summary['acc_labels']),
                                                                                                                f1_all,
                                                                                                                f1_pos,
                                                                                                                precision,
                                                                                                                precision_avg,
                                                                                                                recall,
                                                                                                                recall_avg
                                                                                                                ))

@torch.inference_mode()
def test_narce_iterative(model, data_loader, criterion, criterion2=nn.CrossEntropyLoss(), device='cpu'):
    ''' The model's no longer provided with ground-truth current state s_t at inference time - 
        Instead, it needs to predict current state s'_t at each time t by itself, and use  s'_t to predict CE label y_t and the next state s'_t+1 iteratively.'''
    model.eval()
    model.to(device)
    summary = {'loss': [] , 'acc_labels': [], 'acc_states': []}

    all_labels_pred = []
    all_labels = []
    all_states_pred = []
    all_states = []

    T = data_loader.dataset.data.shape[1]

    for _, batch in enumerate(tqdm(data_loader)):
        in_states = 0
        for i in range(T):
            data = batch[0].to(device)
            labels = batch[1]
            out_states = batch[3]
            
            if i == 0:
                in_states = batch[2].to(device)
                in_states = in_states[:, i:i+1]
            # otherwise, use states predicted in the last iteration
            
            data = data[:, :i+1]

            # Run the Net
            x, s = model(data, in_states)
            
            # Update the next input state by shifting the predicted output state
            _, spred = s.data.topk(1, dim=-1)
            spred = spred.squeeze(-1)
            in_states = torch.cat([in_states, spred[:, i:i+1]], dim=1)

            # Calculate validation loss and accuracy after the target sequence prediction is complete
            if i == T - 1:
                x = x.transpose(-1,1).cpu()
                s = s.transpose(-1,1).cpu()

                criterion = criterion.to('cpu')
                loss = criterion(x, labels)
                state_loss = criterion2(s, out_states)
                loss = 0.5 * loss + 0.5 * state_loss
                summary['loss'].append(loss.item())

                _, xpred = x.data.topk(1, dim=1)
                xpred = xpred.squeeze(1)
                xacc = torch.sum(xpred == labels)/(x.shape[0] * x.shape[-1])
                summary['acc_labels'].append(xacc.item())
                
                _, spred = s.data.topk(1, dim=1)
                spred = spred.squeeze(1)
                sacc = torch.sum(spred == out_states)/(s.shape[0] * s.shape[-1])
                summary['acc_states'].append(sacc.item())

                # Visualize the result
                for j in range(len(xpred)):
                    print("Pred label:",xpred[j])
                    print("True label:", labels[j])
                    print("Pred state:",spred[j])
                    print("True state:", out_states[j])

                all_labels_pred.append(xpred.reshape(-1))
                all_labels.append(labels.reshape(-1))
                all_states_pred.append(spred.reshape(-1))
                all_states.append(out_states.reshape(-1))

    all_labels_pred = np.concatenate(all_labels_pred)
    all_labels = np.concatenate(all_labels)

    f1_all = f1_score(all_labels, all_labels_pred, average='macro')
    f1_pos = f1_score(all_labels, all_labels_pred, labels=[1,2,3], average='macro')

    precision = precision_score(all_labels, all_labels_pred, average=None)
    recall = recall_score(all_labels, all_labels_pred, average=None)
    precision_avg = precision_score(all_labels, all_labels_pred, average='macro')
    recall_avg = recall_score(all_labels, all_labels_pred, average='macro')

    print('Total loss: {}'.format(np.mean(summary['loss'])))
    print('CE labels - Accuracy: {}, F1_all: {}, F1_positive: {}, \n\tPrecision: {}, Avg_P: {}, \n\tRecall: {}, Avg_R: {}'.format(
                                                                                                                np.mean(summary['acc_labels']),
                                                                                                                f1_all,
                                                                                                                f1_pos,
                                                                                                                precision,
                                                                                                                precision_avg,
                                                                                                                recall,
                                                                                                                recall_avg
                                                                                                                ))
    all_states_pred = np.concatenate(all_states_pred)
    all_states = np.concatenate(all_states)
    f1 = f1_score(all_states, all_states_pred, average='macro')

    precision = precision_score(all_states, all_states_pred, average=None)
    recall = recall_score(all_states, all_states_pred, average=None)
    precision_avg = precision_score(all_states, all_states_pred, average='macro')
    recall_avg = recall_score(all_states, all_states_pred, average='macro')
    print('FSM states - Accuracy: {}, F1: {}, \n\tPrecision: {}, Avg_P: {}, \n\tRecall: {}, Avg_R: {}'.format(
                                                                                                        np.mean(summary['acc_states']),
                                                                                                        f1,
                                                                                                        precision,
                                                                                                        precision_avg,
                                                                                                        recall,
                                                                                                        recall_avg
                                                                                                        ))