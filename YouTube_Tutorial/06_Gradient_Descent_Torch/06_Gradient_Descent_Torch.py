"""
    1st Step - Design Model (imput, output, forward pass)
    2nd Step - Construct Loss and Optimizer
    3rd Step - Training Loop:
        - Forward pass: compute prediction
        - Backward pass: gradients
        - Update weights
"""

import torch
import torch.nn as nn #neural network



""" 
    First Model: Manual Prediction
    f = w*x
"""
X=torch.tensor([1,2,3,4], dtype=torch.float32)
Y=torch.tensor([2,4,6,8], dtype=torch.float32)

w=torch.tensor(0.0, dtype=torch.float32, requires_grad=True)

def forward(x):
    return w*x

print(f"Prediction before training: f(5) = {forward(5):.3f}")    

#training
learning_rate=0.001
n_iters=10000

""" 
    Torch Functions:
    - nn.MSELoss() => Mean Squared Error Loss
    - torch.optim.SGD() => Stochastic Gradient Descent
"""
loss=nn.MSELoss() 
optimizer=torch.optim.SGD([w], lr=learning_rate) 

for epoch in range(n_iters):
    #prediction = forward pass
    y_pred=forward(X)
    
    #loss
    l=loss(Y, y_pred)
    
    #gradients = backward pass
    l.backward() #dl/dw
    
    #update weights
    optimizer.step()
    
    #zero gradients
    optimizer.zero_grad()
    
    if epoch % 10 == 0:
        print(f"epoch {epoch+1}: w={w:.3f}, loss={l:.8f}")

print(f"Prediction after training: f(5) = {forward(5):.3f}")


""" 
    Second Model: Pytorch Function
    nn.Linear()
"""
X=torch.tensor([[1],[2],[3],[4]], dtype=torch.float32)
Y=torch.tensor([[2],[4],[6],[8]], dtype=torch.float32)

X_test=torch.tensor([5], dtype=torch.float32)
n_samples, n_features=X.shape
print(f"#samples: {n_samples}, #features: {n_features}")


##Define the model
imput_size=n_features
output_size=n_features
#model=nn.Linear(imput_size, output_size) #This works only with single-neuron model

"""
    A structurad way to define the model, using nn.Module.
    It is mandatory for model bigger than a single neuron
"""
class LinearRegression(nn.Module):
    def __init__(self, imput_dim, output_dim):
        super(LinearRegression, self).__init__() #super() is used to call the __init__() of the parent class, in this case nn.Module
        #define layers
        self.lin=nn.Linear(imput_dim, output_dim)
        
    def forward(self, x):
        return self.lin(x)
    
model=LinearRegression(imput_size, output_size)

print(f"Prediction before training: f(5) = {model(X_test).item():.3f}")    #Tensor.item() return the actual value

#training
learning_rate=0.01
n_iters=100

""" 
    Torch Functions:
    - nn.MSELoss() => Mean Squared Error Loss
    - torch.optim.SGD() => Stochastic Gradient Descent
"""
loss=nn.MSELoss() 
optimizer=torch.optim.SGD(model.parameters(), lr=learning_rate) 

for epoch in range(n_iters):
    #prediction = forward pass
    y_pred=model(X)
    
    #loss
    l=loss(Y, y_pred)
    
    #gradients = backward pass
    l.backward() #dl/dw
    
    #update weights
    optimizer.step()
    
    #zero gradients
    optimizer.zero_grad()
    
    if epoch % 10 == 0:
        [w,b]=model.parameters()
        print(f"epoch {epoch+1}: w={w[0][0].item():.3f}, loss={l:.8f}")

print(f"Prediction after training: f(5) = {model(X_test).item():.3f}")



