import torch
import numpy as np

#empty tensor
x=torch.empty(2,1)
print(x)

#random tensor
x=torch.randn(2,1)
print(x)

#ones tensor
x=torch.ones(2,2, dtype=torch.float32)
print(x.type())

#basic operations
x=torch.rand(2,2)
y=torch.rand(2,2)
print(x)
print(y)
y.add_(x)  #evry operation with "_" is a replacment one
print(y)



#reshape tensor
print("\nReshaping Tensor")
x=torch.rand(4,4)
print("Original Tensor\n",x)
y=x.view(16)
print("Reshaped Tensor\n",y)
z=x.view(-1,8)  #with -1, you have to specify the second numer and it adjusts the first one
print("Reshaped Tensor, Automatic Way\n",z)


#convert torch tensor to numpy array
a=torch.ones(5)
print(a)
b=a.numpy()
print(type(b))


#convert numpy array to torch tensor
a=np.ones(5)
print(a)
b=torch.from_numpy(a)
print(b.type)

#creating a tensor on GPU
if torch.cuda.is_available():
    device=torch.device("cuda")
    x=torch.ones(5,device=device)
    y=torch.ones(5)
    y=y.to(device)
    z=x+y
    z=z.to("cpu",torch.double)
    print(z)
else:
    print("CUDA is not available")



