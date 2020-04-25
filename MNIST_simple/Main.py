import torch
import time
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
import torchvision
import matplotlib.pyplot as plt
from tqdm import tqdm as tqdm
cuda = torch.device('cuda')
print(torch.cuda.set_device(0), torch.cuda.current_device(), torch.cuda.get_device_name(0))

dl = DataLoader(torchvision.datasets.MNIST('/data/mnist', train=True, download=True))

x = dl.dataset.data
x = x.to(dtype=torch.float32)
x = torch.FloatTensor(x)
#x = x.to(device=cuda)
x = x.reshape([1875,32,1,28,28])
y = dl.dataset.targets
y = y.to(dtype=torch.long)
y = torch.LongTensor(y)
y = y.reshape([1875,32])
#y = y.to(device=cuda)
class CNN_MNIST(nn.Module):
    def __init__(self):
        super(CNN_MNIST,self).__init__()
        self.conv1 = nn.Conv2d(1,48,4)
        self.conv2 = nn.Conv2d(48,96,2)
        self.maxpool=nn.MaxPool2d([(28-4),(28-4)])
        self.dense1= nn.Linear(96,512)
        self.dense2= nn.Linear(512,10)
    def forward(self, x):
        x = torch.sigmoid(self.conv1(x))
        x = torch.sigmoid(self.conv2(x))
        x = self.maxpool(x)
        x = torch.flatten(x,1,-1)
        x = self.dense1(x)
        x = torch.softmax(self.dense2(x),0)
        return x
log = []
model = CNN_MNIST().cuda(0)

try:
    model.load_state_dict(torch.load("mnist.mod"))

except:
    print("\ncreating new state-dict")


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adadelta(model.parameters(),lr=0.01,rho=0.9)

def train(crit,optim,model,x,y,epochs,log):
    x = x.to(device=cuda)
    y = y.to(device=cuda)
    for epoch in range(epochs):
        print("\nepoch:", epoch+1,"of",epochs)
        time.sleep(.1)
        for i in tqdm(range(len(x))):
                optim.zero_grad()
                output = model.forward(x[i])
                loss = crit(output,y[i])
                log.append(loss.item())
                loss.backward()
                optim.step()

    return log

def test(model,x,y):
    print("Now testing...")
    time.sleep(.1)
    acc = 0
    out = []
    out=model.forward(x[0])
    for i in tqdm(range(len(x))):
       out=(model.forward(x[i]))
       if out==y[i]:
            acc+=1
    return acc/len(x)

epochs=100

#model_cpu=CNN_MNIST()
#model_cpu.load_state_dict(torch.load("mnist.mod"))
#acc_bef=test(model_cpu,x,y)

log=train(criterion,optimizer,model,x,y,epochs,log)
x_val=np.arange(len(log))
plt.plot(x_val,np.array(log))
plt.show()
#acc_curr=test(model_cpu,x,y)
#model_cpu=CNN_MNIST()
#model_cpu.load_state_dict(torch.load("mnist.mod"))
#print("\nPrevious accuaracy:",acc_bef,"\nCurrent accuaracy:",acc_curr,"\nrelative change:",acc_curr-acc_bef)

torch.save(model.state_dict(),'./mnist.mod')