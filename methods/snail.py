import math
import torch
import torch.nn as nn
import wandb

from blocks import *
from methods.meta_template import MetaTemplate

class SNAIL(MetaTemplate):
    def __init__(self, backbone, n_way, n_support):
        # N-way, K-shot
        super(SNAIL, self).__init__(backbone, n_way, n_support, change_way=False)
        
        self.loss_fn = nn.CrossEntropyLoss()
        T = n_way * n_support + 1
        
        num_channels = self.feat_dim + n_way # (input_dim + one_hot_dim)
        num_filters = int(math.ceil(math.log(T, 2)))
        self.attention1 = AttentionBlock(num_channels, 64, 32)
        
        num_channels += 32
        self.tc1 = TCBlock(num_channels, T, 128)
        num_channels += num_filters * 128
        self.attention2 = AttentionBlock(num_channels, 256, 128)
        
        num_channels += 128
        self.tc2 = TCBlock(num_channels, T, 128)
        num_channels += num_filters * 128
        self.attention3 = AttentionBlock(num_channels, 512, 256)
        
        num_channels += 256
        self.fc = nn.Linear(num_channels, n_way)
        
    def get_label_map(labels):
        # Create a map from labels to indexes
        labels = labels.numpy()
        unique = np.unique(labels)
        map = {label:idx for idx, label in enumerate(unique)}
        
        return map
    
    def get_one_hots(labels, map):
        # Map labels to their one-hot representations
        labels = labels.numpy()
        idxs = [map[labels[i]] for i in range(labels.size)]
        one_hot = np.zeros((labels.size, len(map)))
        one_hot[np.arange(labels.size), idxs] = 1
        
        return one_hot, idxs

    def forward(self, input, labels):
        x = self.encoder(input)
        batch_size = int(labels.size()[0] / (self.n_way * self.n_support + 1))
        last_idxs = [(i + 1) * (self.n_way * self.n_support + 1) - 1 for i in range(batch_size)]
        labels[last_idxs] = torch.Tensor(np.zeros((batch_size, labels.size()[1]))).to(self.device)
        
        x = torch.cat((x, labels), 1)
        x = x.view((batch_size, self.n_way * self.n_support + 1, -1))
        x = self.attention1(x)
        x = self.tc1(x)
        x = self.attention2(x)
        x = self.tc2(x)
        x = self.attention3(x)
        x = self.fc(x)
        
        return x
    
    def set_forward(self, x, y=None):
        sequences = []
        # get support set and labels
        support_set = x[:, :self.n_support, :] # N x K samples that are fixed for each sequence
        support_set = support_set.contiguous().view(-1, x.shape[2]) # flatten to get (N * K, input_dim)
        support_labels = y[:, :self.n_support].flatten() # N * K labels that are fixed for each sequence

        # get label map
        label_map = self.get_label_map(support_labels)

        all_labels = []
        pred_targets = []
        n_query = x.shape[1] - self.n_support
        for i in range(n_query):
            for j in range(self.n_way):
                query_sample = x[j, self.n_support + i, :].unsqueeze(0) # get a new query sample
                seq = torch.cat((support_set, query_sample), dim=0) # sequence of N x K + 1 samples
                sequences.append(seq) # append to list of sequences
                
                # Get labels one-hot representations and indexes
                last_seq_label = y[j, self.n_support + i]
                seq_labels = torch.cat((support_labels, torch.Tensor([last_seq_label]).long())) # N x K + 1 labels
                labels, idxs = self.get_one_hots(seq_labels, label_map)

                all_labels.append(labels)
                pred_targets.append(idxs[-1])

        # sequences are of shape (n_query * n_cls, (N * K + 1), input_dim))
        # all_labels are of shape (n_query * n_cls, (N * K + 1), num_cls)
        # pred_tagets are of shape (n_query * n_cls)

        labels = torch.Tensor(all_labels).view(-1, self.n_way)
        pred_targets = torch.Tensor(pred_targets).long()
        
        sequences = torch.stack(sequences)
        sequences = sequences.view(-1, x.shape[2])
        
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
