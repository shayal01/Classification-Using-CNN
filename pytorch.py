# -*- coding: utf-8 -*-
"""
Created on Sat Jul  2 23:15:48 2022

@author: shaya
"""
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader , TensorDataset

from sklearn.preprocessing import OneHotEncoder
from scipy import io

class SVHN_classifier(nn.Module):
    def __init__(self,num_classes):
        super().__init__()
        self.conv1=nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(5,5),
                               stride=1,padding='same')
        self.mpool=nn.MaxPool2d(kernel_size=2,stride=2)
        self.conv2=nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(5,5),
                               stride=1,padding='same')
        self.conv3=nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(5,5),
                               stride=1,padding='same')
        self.flat=nn.Flatten()
        self.fc1=nn.Linear(in_features=128*8*8,out_features=3072)
        self.fc2=nn.Linear(in_features=3072,out_features=2048)
        self.fc3=nn.Linear(in_features=2048,out_features=10)
    

    def forward(self,x):
        x=F.relu(self.conv1(x))
        x=self.mpool(x)
        x=F.relu(self.conv2(x))
        x=self.mpool(x)
        x=F.relu(self.conv3(x))
        x=self.flat(x)
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=F.softmax(self.fc3(x))
        return x


def Norm(data):
    transform=transforms.ToTensor()
    image=torch.empty((data.shape[0],3,32,32))
    for i in  range(data.shape[0]):
        image[i]=transform(data[i])
        
    return image       

def one_hot(label):
    
    onehot_encoder = OneHotEncoder(sparse=False)
    onehot_encoded = onehot_encoder.fit_transform(label)
    label=torch.from_numpy( onehot_encoded )
   
    return label
    



EPOCHS=2
BATCH_SIZE=128
NUM_CLASSES=10



trX=io.loadmat(r'C:\Users\shaya\Downloads\train_32x32.mat')['X']
trY = io.loadmat(r'C:\Users\shaya\Downloads\train_32x32.mat')['y']
tsX = io.loadmat(r'C:\Users\shaya\Downloads\test_32x32.mat')['X']
tsY = io.loadmat(r'C:\Users\shaya\Downloads\test_32x32.mat')['y']



trX_norm=Norm(trX)
tsX_norm=Norm(tsX)

trY_hot=one_hot(trY)
tsY_hot=one_hot(tsY)

train_set=TensorDataset(trX_norm,trY_hot)
train_loader=DataLoader(train_set,batch_size=BATCH_SIZE,shuffle=True)

test_set=TensorDataset(tsX_norm,tsY_hot)
test_loader=DataLoader(test_set,batch_size=BATCH_SIZE,shuffle=True)

svhn_model=SVHN_classifier(NUM_CLASSES)
optimizer=optim.SGD(svhn_model.parameters(),lr=0.01)
criterion=nn.CrossEntropyLoss()

num_batches=len(train_loader)

for epoch in range(EPOCHS):
    for batch_idx,(images,labels) in enumerate(train_loader):
        optimizer.zero_grad()
        output=svhn_model(images)
        loss=criterion(output,labels)
        loss.backward()
        optimizer.step()
        
        if batch_idx % 100 == 0:               # report periodically
            batch_loss = loss.mean().item()
            print("Epoch {}/{}\tBatch {}/{}\tLoss: {}" \
                    .format(epoch, EPOCHS, batch_idx, num_batches, batch_loss))



num_correct=0
nm_attempts=0

for images,labels in test_loader:
    with torch.no_grad():
        output=svhn_model(images)
        
