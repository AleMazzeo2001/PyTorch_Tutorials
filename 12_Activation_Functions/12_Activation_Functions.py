import torch
import torch.nn as nn
import torch.nn.functional as F # functions which don't have any parameters


#nn.ReLU() creates an nn.Module which you can add e.g. to an nn.Sequential model.
#torch.relu on the other side is just the functional API call to the relu function,
#so that you can add it e.g. in your forward method yourself.

# option 1: create nn modules => nn.ReLU()

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(NeuralNet, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size) 
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, 1)  
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # F.leaky_relu # leaky_relu, this is not in nn
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.sigmoid(out)
        return out
    
# option 2: use activation function directly in forward method => torch.relu()

class NeuralNet2(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(NeuralNet2, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size) 
        self.linear2 = nn.Linear(hidden_size, 1)  
    
    def forward(self, x):
        out = torch.relu(self.linear1(x))
        out = torch.sigmoid(self.linear2(out))
        return out
    



x = torch.tensor([-1.0, 1.0, 2.0, 3.0])

# sofmax
print('\nsoftmax')
output = torch.softmax(x, dim=0)
print(f"torch.function: {output}")
sm = nn.Softmax(dim=0)
output = sm(x)
print(f"nn.Module: {output}")


# sigmoid 
print('\nsigmoid')
output = torch.sigmoid(x)
print(f"torch.function: {output}")
s = nn.Sigmoid()
output = s(x)
print(f"nn.Module: {output}")

#tanh
print('\ntanh')
output = torch.tanh(x)
print(f"torch.function: {output}")
t = nn.Tanh()
output = t(x)
print(f"nn.Module: {output}")

# relu
print('\nrelu')
output = torch.relu(x)
print(f"torch.function: {output}")
relu = nn.ReLU()
output = relu(x)
print(f"nn.Module: {output}")

# leaky relu
print('\nleaky relu')
output = F.leaky_relu(x)
print(f"torch.function: {output}")
lrelu = nn.LeakyReLU()
output = lrelu(x)
print(f"nn.Module: {output}")
    