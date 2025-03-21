"""
    1) Designe Model (input, output size, forward pass)
    2) Construct loss and optimizer
    3) Training loop:
        - Forward pass: compute prediction and loss
        - Backward pass: gradients
        - Update weights
"""

import torch
import torch.nn as nn
import numpy as np  
from sklearn import datasets
import matplotlib.pyplot as plt


# 0) Prepare data
X_numpy, y_numpy = datasets.make_regression(n_samples=500, n_features=1, noise=10, random_state=1) #generate random data starting from a linear regression model

X = torch.from_numpy(X_numpy.astype(np.float32))
y = torch.from_numpy(y_numpy.astype(np.float32))
y = y.view(y.shape[0], 1) #reshape y to fit the model

n_samples, n_features=X.shape
print(f"#samples: {n_samples}, #features: {n_features}")

# 1) Model
input_size=n_features
output_size=1

model=nn.Linear(input_size, output_size)


# 2) Loss and optimizer
criterion=nn.MSELoss() 
optimizer=torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.99) #momentum is a parameter that helps to avoid local minimas

# 3) Training loop
num_epochs=1000
losses=[]
epochs=[]

plt.ion()
fig, ax = plt.subplots()
line, = ax.plot(epochs, losses, 'r-')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.set_title('Training Loss')

for epoch in range(num_epochs):
    # forward pass
    y_pred = model(X)
    loss = criterion(y_pred, y)


    # backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    epochs.append(epoch + 1)
    losses.append(loss.item())


    # Update plot
    if epoch % 25 == 0:
        print(f'Epoch {epoch+1}: Loss = {loss.item():.8f}')
        line.set_xdata(epochs)
        line.set_ydata(losses)
        ax.relim()
        ax.autoscale_view()
        ax.grid()
        plt.draw()
        plt.pause(0.02)

plt.ioff()
plt.grid()
plt.show()

# Final plot
preticted = model(X).detach().numpy()
plt.plot(X_numpy, y_numpy, 'ro')
plt.plot(X_numpy, preticted, 'b')
plt.grid()
plt.title(f"Final Prediction\n MSE={losses[-1]:.5f}")
plt.show()
