import math
import torch
import torch.nn as nn
import wandb

from methods.blocks import *
from methods.meta_template import MetaTemplate

class SNAIL(MetaTemplate):
    def __init__(self, backbone, n_way, n_support):
        super(SNAIL, self).__init__(backbone, n_way, n_support, change_way=False)
        
        self.loss_fn = nn.CrossEntropyLoss()
        T = n_way * n_support + 1 # N * K + 1
        
        layers = []
        num_channels = self.feat_dim + n_way # (input_dim + one_hot_dim)
        num_filters = int(math.ceil(math.log(T, 2)))
        layers.append(AttentionBlock(num_channels, 64, 32))
        
        num_channels += 32
        layers.append(TCBlock(num_channels, T, 128))
        num_channels += num_filters * 128
        layers.append(AttentionBlock(num_channels, 256, 128))
        
        num_channels += 128
        layers.append(TCBlock(num_channels, T, 128))
        num_channels += num_filters * 128
        layers.append(AttentionBlock(num_channels, 512, 256))
        
        num_channels += 256
        layers.append(nn.Linear(num_channels, n_way))
        
        # Build layers
        self.layers = nn.Sequential(*layers)
        
    def labels_to_one_hot(self, labels):
        # Create a map from labels to indices
        labels = labels.numpy()
        unique = np.unique(labels)
        map = {label:idx for idx, label in enumerate(unique)}
        
        # Create the one-hot encoding based on the indices
        idxs = [map[labels[i]] for i in range(labels.size)]
        one_hot = np.zeros((labels.size, len(map)))
        one_hot[np.arange(labels.size), idxs] = 1
        
        return one_hot, idxs

    def forward(self, input, labels):
        # Get the embeddings
        x = self.feature(input)
        
        # Zero out the last label of each sequence
        batch_size = int(labels.size()[0] / (self.n_way * self.n_support + 1))
        last_idxs = [(i + 1) * (self.n_way * self.n_support + 1) - 1 for i in range(batch_size)]
        labels[last_idxs] = torch.Tensor(np.zeros((batch_size, labels.size()[1]))).to(self.device)
        
        # Append the corresponding label to each sample
        x = torch.cat((x, labels), 1)
        x = x.view((batch_size, self.n_way * self.n_support + 1, -1))
        
        # Forward to the model
        x = self.layers(x)
        
        return x
    
    def set_forward(self, x, y=None):
        sequences = []
        # Get support set and labels
        support_set = x[:, :self.n_support, :] # N x K samples that are fixed for each sequence
        support_set = support_set.contiguous().view(-1, x.shape[2]) # flatten to get (N * K, input_dim)
        support_labels = y[:, :self.n_support].flatten() # N * K labels that are fixed for each sequence

        all_labels = []
        pred_targets = []
        n_query = x.shape[1] - self.n_support
        # For every query sample, build the corresponding N*K + 1 sequence
        for i in range(n_query):
            for j in range(self.n_way):
                # Shuffle the support set for every sequence
                indices = torch.randperm(support_set.shape[0])
                sup_set = support_set[indices]
                sup_labels = support_labels[indices]
                
                # Get a new query sample
                query_sample = x[j, self.n_support + i, :].unsqueeze(0)
                # Append it to its support set
                seq = torch.cat((sup_set, query_sample), dim=0)
                # Append the sequence to the list of sequences
                sequences.append(seq)
                
                # Get a new query label
                last_seq_label = y[j, self.n_support + i]
                # Append it to the support labels
                seq_labels = torch.cat((sup_labels, torch.Tensor([last_seq_label]).long())) # N x K + 1 labels
                # Get one-hot representations of the labels and their indices (this becomes the targets)
                labels, idxs = self.labels_to_one_hot(seq_labels)

                all_labels.append(labels)
                # We only want to predict the last label
                pred_targets.append(idxs[-1])

        # sequences are of shape (n_query * n_cls, (N * K + 1), input_dim))
        # all_labels are of shape (n_query * n_cls, (N * K + 1), num_cls)
        # pred_tagets are of shape (n_query * n_cls)

        # Set shape to (n_query * n_cls * (N * K + 1), num_cls) to further concatenate each embedding with its label
        labels = torch.Tensor(np.array(all_labels)).view(-1, self.n_way).to(self.device)
        pred_targets = torch.Tensor(np.array(pred_targets)).long().to(self.device)
        
        # Set shape to (n_query * n_cls * (N * K + 1), input_dim) to further concatenate each embedding with its label
        sequences = torch.stack(sequences)
        sequences = sequences.view(-1, x.shape[2]).to(self.device)
        
        # Forward to the model
        output = self.forward(sequences, labels)
        # Get outputs for last sample
        scores = output[:, -1, :]
        
        return scores, pred_targets
    
    def set_forward_loss(self, x, y=None):
        scores, pred_targets = self.set_forward(x, y)
        loss = self.loss_fn(scores, pred_targets)

        return loss
    
    def train_loop(self, epoch, train_loader, optimizer):
        print_freq = 10

        avg_loss = 0
        for i, (x, y) in enumerate(train_loader):
            if isinstance(x, list):
                self.n_query = x[0].size(1) - self.n_support
            else: 
                self.n_query = x.size(1) - self.n_support
                
            optimizer.zero_grad()
            loss = self.set_forward_loss(x, y)
            loss.backward()
            optimizer.step()
            avg_loss = avg_loss + loss.item()

            if i % print_freq == 0:
                print('Epoch {:d} | Batch {:d}/{:d} | Loss {:f}'.format(epoch, i, len(train_loader),
                                                                        avg_loss / float(i + 1)))
                wandb.log({"loss": avg_loss / float(i + 1)})
                
    def correct(self, x, y=None):
        scores, pred_targets = self.set_forward(x, y)
        y_query = pred_targets.cpu().numpy().tolist()

        topk_scores, topk_labels = scores.data.topk(1, 1, True, True)
        topk_ind = topk_labels.cpu().numpy()
        top1_correct = np.sum(topk_ind[:, 0] == y_query)
        return float(top1_correct), len(y_query)
    
    def test_loop(self, test_loader, record=None, return_std=False):
        correct = 0
        count = 0
        acc_all = []

        iter_num = len(test_loader)
        for i, (x, y) in enumerate(test_loader):
            if isinstance(x, list):
                self.n_query = x[0].size(1) - self.n_support
                if self.change_way:
                    self.n_way = x[0].size(0)
            else: 
                self.n_query = x.size(1) - self.n_support
                if self.change_way:
                    self.n_way = x.size(0)
            correct_this, count_this = self.correct(x, y)
            acc_all.append(correct_this / count_this * 100)

        acc_all = np.asarray(acc_all)
        acc_mean = np.mean(acc_all)
        acc_std = np.std(acc_all)
        print('%d Test Acc = %4.2f%% +- %4.2f%%' % (iter_num, acc_mean, 1.96 * acc_std / np.sqrt(iter_num)))

        if return_std:
            return acc_mean, acc_std
        else:
            return acc_mean
