import torch

torch.manual_seed(27029)
x=torch.tensor([1.0], dtype=torch.float32, requires_grad=False)
y=torch.tensor([2.0], dtype=torch.float32, requires_grad=False)
w=torch.tensor([1.0], dtype=torch.float32, requires_grad=True)

#forward pass and comupte the loss
y_hat=w*x
loss=(y_hat-y)**2  #RSS loss

print(f"Loss: \n{loss}")

#backward pass
loss.backward()
print(f"Gradient: \n{w.grad}")
