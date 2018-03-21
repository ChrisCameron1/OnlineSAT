from __future__ import print_function

import glob
from tqdm import tqdm

from scipy.sparse import csr_matrix
import numpy as np
from torch.utils.data import Dataset
import torch.nn as nn
import torch
from torch.autograd import Variable
from torch.nn.init import xavier_uniform, normal

import exchangable_tensor
from exchangable_tensor.sp_layers import mean_pool, SparsePool, SparseExchangeable, SparseSequential

from data import prep

batch_size = 100
epochs = 100000
out_dim = 32

def prep_data(x, requires_grad=False, use_cuda=True):
    '''
    Helper function for setting up data variables
    '''
    x = Variable(x, requires_grad=requires_grad)
    if use_cuda:
        x = x.cuda()
    return x


class SATDataset(Dataset):
    '''
    A dataset object for holding 
    '''
    def __init__(self, values, index, y_sat, y_unsat):
        self.index = np.array(index, dtype="int")
        self.y_sat = np.array(y_sat, dtype="int")
        self.y_unsat = np.array(y_unsat, dtype="int")
        self.values = np.array(values, dtype="float32").reshape(index.shape[0], -1) # ensure 2D

    def __len__(self):
        return self.index.shape[0]

    def __getitem__(self, index):
        return {"index": self.index[index, :], 
                "input": self.values[index, :], 
                "target": self.y_sat[index],
                "unsat": self.y_unsat[index]}

def load_data(path, validation=0., seed=None):
    data = np.load(path)
    values = data['values']
    indices = data['indices']
    y_sat = data['y_sat']
    y_unsat = data['y_unsat']
    return SATDataset(values, indices, y_sat, y_unsat)

def is_sat(dat, y):
    X_sp = csr_matrix((np.dot(dat.values, np.array([[1], [-1]])).flatten(), 
                      (dat.index[:, 0],dat.index[:, 1])), 
                      shape=dat.index.max(axis=0) + 1)
    if not isinstance(y, np.ndarray):
        y = y.cpu().data.numpy().flatten()
    y_hat = y.round()
    y_hat[y_hat == 0] = -1
    return (X_sp.dot(y_hat) == -3).sum(axis=0) == 0
    
files = glob.glob("data/sat*")

dat = load_data(files[0])

index = prep(dat.index, dtype="int")
index = index.cuda()
enc = SparseSequential(index, 
                       SparseExchangeable(2,250, index), 
                       nn.LeakyReLU(),
                      # torch.nn.Dropout(p=0.5),
                       SparseExchangeable(250,250, index),
                       nn.LeakyReLU(),
                       SparseExchangeable(250,150, index),
                       nn.LeakyReLU(),
                       #torch.nn.Dropout(p=0.5),
                       SparseExchangeable(150,150, index),
                       nn.LeakyReLU(),
                       torch.nn.Dropout(p=0.5),
                       SparseExchangeable(150,64, index),
                       nn.LeakyReLU(),
                       torch.nn.Dropout(p=0.5),
                       SparseExchangeable(64,32, index),
                       nn.LeakyReLU(),
                       torch.nn.Dropout(p=0.5),
                       SparseExchangeable(32,1, index),
                       SparsePool(index, 1, 1, keep_dims=False)
                   )

mod = torch.nn.Sequential(enc,
                         # torch.nn.Linear(in_features=out_dim, out_features=16),
                         # nn.LeakyReLU(),
                         # torch.nn.Dropout(p=0.5),
                         # torch.nn.Linear(in_features=16, out_features=1),
                          nn.Sigmoid())

# as direct access to tensors data attribute
def weights_init(m):
    if isinstance(m, nn.Linear):
        xavier_uniform(m.weight.data)

mod.apply(weights_init)
mod.cuda()
loss = nn.BCELoss()
optimizer = torch.optim.Adam(mod.parameters(), lr=0.001)

for epoch in xrange(epochs):
    mod.train()
    optimizer.zero_grad()
    loss_batch = 0.
    av_acc = 0.
    for _ in tqdm(xrange(batch_size)):
        i = np.random.randint(0, int(len(files) * 0.9))
        dat = load_data(files[i])
        input = prep_data(torch.from_numpy(dat.values))
        index = prep_data(torch.from_numpy(dat.index))
        enc.index = index
        input = input.cuda()
        y_sat = dat.y_sat[:,0]
        y_sat[y_sat == -1] = 0
        target = prep_data(torch.from_numpy(y_sat))
        target = target.cuda()
        out = mod(input).squeeze()
        loss_i = loss(out, target.float())
        loss_batch = loss_batch + loss_i
        acc = np.mean(np.abs(np.round(out.cpu().data.numpy()).flatten() - y_sat))
        av_acc += acc
    av_acc = av_acc / (1. * batch_size)
    loss_batch = loss_batch / (1.* batch_size)
    if epoch % 1 == 0:
        p_loss = loss_batch.cpu().data.numpy()[0]
        print("Epoch: {ep}. Batch loss: {loss}. Accuracy: {acc}".format(ep=epoch, 
                                                                        loss=p_loss, 
                                                                        acc=1.-av_acc))
        #perf.append(p_loss)
    loss_batch.backward()
    optimizer.step()
    
    if epoch % 50 == 0:
        mod.eval()
        # evaluate
        loss_batch = 0.
        av_acc = 0.
        n_sat = 0
        n_eval = float(len(files) - int(len(files) * 0.9))
        for i in tqdm(xrange(int(len(files) * 0.9), len(files))):
            dat = load_data(files[i])
            input = prep_data(torch.from_numpy(dat.values))
            index = prep_data(torch.from_numpy(dat.index))
            enc.index = index
            input = input.cuda()
            y_sat = dat.y_sat[:,0]
            y_sat[y_sat == -1] = 0
            target = prep_data(torch.from_numpy(y_sat))
            target = target.cuda()
            out = mod(input).squeeze()
            loss_i = loss(out, target.float()).cpu().data.numpy()[0]
            loss_batch = loss_batch + loss_i
            acc = np.mean(np.abs(np.round(out.cpu().data.numpy()).flatten() - y_sat))
            av_acc += acc
            n_sat += int(is_sat(dat, out))
        n_sat = n_sat / n_eval
        av_acc = av_acc / n_eval
        v_loss = loss_batch / n_eval
        print("EVALUATION. Epoch: {ep}. Batch loss: {loss}. Accuracy: {acc}. Percent SAT: {sat}".format(ep=epoch, 
                                                                        loss=v_loss, 
                                                                        acc=1.-av_acc,
                                                                        sat=n_sat))
        print("{ep}, {loss},{t_loss},{acc},{sat}".format(ep=epoch, loss=v_loss, 
                                                     t_loss=p_loss, acc=1.-av_acc,sat=n_sat),
             file=open("perf2.csv", "a"))
        torch.save(mod, 'checkpts/model2_ep_%06d.pt' % epoch)

        
        
        