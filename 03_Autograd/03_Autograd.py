import torch
import numpy as np

""" 
    This script demonstrates PyTorch's autograd functionality, 
    showcasing dynamic computation graph creation and automatic gradient calculation 
    for optimizing machine learning models.
"""

torch.manual_seed(27029)

""" 
    "requires_grad=True" is used to track computation of the gradient in a numeric way
    "requires_grad=False" does not track computation of the gradient
"""
x=torch.randn(3, requires_grad=True)
print(x)
# output: tensor([-0.3213, -0.7387,  0.0674], requires_grad=True)

"""
    "grad_fn" tells us information about the operation that created the tensor
    It is a way to trace the tensor back to its origin
"""
##Addition => grad_fn=<AddBackward0>
y=x+2
print(y)
# output: tensor([1.6787, 1.2613, 2.0674], grad_fn=<AddBackward0>)

##Multiplication => grad_fn=<MulBackward0>
z=y*y*2
print(z)
# output: tensor([5.6362, 3.1817, 8.5479], grad_fn=<MulBackward0>)

##Mean => grad_fn=<MeanBackward0>
z=z.mean()  #grad can be implicitly created only for scalar outputs
print(z)
# output: tensor(5.7886, grad_fn=<MeanBackward0>)

"""
    "backward()" is used to calculate the gradient of the tensor 
    with respect to the input tensor with requires_grad=True

    Note: The gradient is accumulated into the .grad attribute of the tensor
    Use "x.grad.zero_()" to clear the gradient before calling backward()
"""
z.backward() #dz/dx => we're computing in a dynamic graph way
print(x.grad) #numerical value of dz/dx having tensor x as input


"""
    If the tensor is not a scalar, we need to pass a gradient argument that is a tensor of matching shape
    This is often used in more complex neural network training scenarios
"""
z=y*y*2
v=torch.tensor([0.1,1.0,0.001], dtype=torch.float32)
z.backward(v) 
print(x.grad) 

"""
    In some operations, such as inference, we don't need the gradient
    
    There are three different ways:
    1. x.requires_grad_(False)
    2. x.detach()
    3. with torch.no_grad():
        do something...
"""
x.requires_grad_(False) # "_" operation: updating the tensor
print(x.requires_grad) #output: False



#Dummy training example
print("\n\n\nDummy training example")
weights=torch.ones(4, dtype=torch.float32, requires_grad=True)  # In Neural Network we must to compute the gradinet of the 
                                           # Loss with respect to the weights => requires_grad=True
for epoch in range(3):
    model_output=(weights*3).sum() #dummy model
    model_output.backward() #dLoss/dWeights
    print(weights.grad)
    weights.grad.zero_() 