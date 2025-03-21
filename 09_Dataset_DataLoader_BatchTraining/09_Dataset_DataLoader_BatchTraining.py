import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math

class WineDataset(Dataset):
    
    def __init__(self):
        # data loading

        xy=np.loadtxt('wine.csv', delimiter=',', dtype=np.float32, skiprows=1)
        self.x=torch.from_numpy(xy[:, 1:]) 
        self.y=torch.from_numpy(xy[:, [0]]) # n_samples, 1
        self.n_samples=xy.shape[0]

    def __getitem__(self, index):
        # dataset[0]
        return self.x[index], self.y[index]

    def __len__(self):
        # len(dataset)
        return self.n_samples
    
dataset=WineDataset()
dataloader=DataLoader(dataset=dataset, batch_size=4, shuffle=True, num_workers=0)


# convert to an iterator and look at one random sample
dataiter = iter(dataloader)
data = next(dataiter)
features, labels = data
#print(features, labels)


# Dummy Training loop
num_epochs = 2
total_samples = len(dataset)
n_iterations = math.ceil(total_samples/4)
print(f"Number of Samples={total_samples}, Number of Iterations={n_iterations}")

for epoch in range(num_epochs):
    for i, (inputs, labels) in enumerate(dataloader):
        # forward, backward, update
        if (i+1)%5==0:
            print(f'epoch {epoch+1}/{num_epochs}, step {i+1}/{n_iterations}, inputs {inputs.shape}')


# some famous datasets are available in torchvision.datasets
# e.g. MNIST, Fashion-MNIST, CIFAR10, COCO

train_dataset = torchvision.datasets.MNIST(root='./data', 
                                           train=True, 
                                           transform=torchvision.transforms.ToTensor(),  
                                           download=True)

train_loader = DataLoader(dataset=train_dataset, 
                                           batch_size=3, 
                                           shuffle=True)

# look at one random sample
dataiter = iter(train_loader)
data = next(dataiter)
inputs, targets = data
print(inputs.shape, targets.shape)

torchvision.datasets.