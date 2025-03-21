import torch
import torch.nn as nn
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt


# 0) Prepare data
bc = datasets.load_breast_cancer() #import the dataset
X,y=bc.data, bc.target #features and target
#print(f"X.shape: {X.shape}, y.shape: {y.shape}")

n_samples, n_features=X.shape
print(f"#samples: {n_samples}, #features: {n_features}")

# split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=27029)
print(f"X_train.shape: {X_train.shape}, X_test.shape: {X_test.shape}")

# scale
sc=StandardScaler() #standardize features by removing the mean and scaling to unit variance
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)
#print(f"X_train_scaled[0].\n{X_train[0]}")

#conversion to torch tensors
X_train=torch.from_numpy(X_train.astype(np.float32))
X_test=torch.from_numpy(X_test.astype(np.float32))
y_train=torch.from_numpy(y_train.astype(np.float32))
y_test=torch.from_numpy(y_test.astype(np.float32))

y_train=y_train.view(y_train.shape[0], 1)
y_test=y_test.view(y_test.shape[0], 1)


# 1) Model f=wx+b, logistic regression at the end
class LogisticRegression(nn.Module):
    def __init__(self, n_input_features):
        super(LogisticRegression, self).__init__()
        self.linear=nn.Linear(n_input_features, 1)
        
    def forward(self, x):
        y_predicted=torch.sigmoid(self.linear(x))
        return y_predicted
    
model=LogisticRegression(n_features)


# 2) Loss and optimizer
criterion = nn.BCELoss() 
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9) # momentum is a parameter that helps to avoid local minimas


# 3) Training loop
num_epochs = 1000
losses = []
epochs = []

plt.ion()
fig, ax = plt.subplots()
line, = ax.plot(epochs, losses, 'r-')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.set_title('Training Loss')

for epoch in range(num_epochs):
    # forward pass
    y_pred = model(X_train)
    loss = criterion(y_pred, y_train)

    # backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    epochs.append(epoch + 1)
    losses.append(loss.item())

    # Debugging prints
    print(f'Epoch {epoch+1}: Loss = {loss.item():.8f}')
   

    # Update plot
    if epoch % 10 == 0:
        line.set_xdata(epochs)
        line.set_ydata(losses)
        ax.relim()
        ax.autoscale_view()
        plt.draw()
        plt.pause(0.01)

plt.ioff()
plt.show()

with torch.no_grad():
    y_predicted=model(X_test)
    y_predicted_cls=y_predicted.round()
    acc=y_predicted_cls.eq(y_test).sum()/float(y_test.shape[0])
    print(f"Test Accuracy = {acc:.8f}")


