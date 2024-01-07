import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import os
import time
import matplotlib.pyplot as plt

class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()
        self.conv1=nn.Conv2d(3,32,3,1)
        self.conv2=nn.Conv2d(32,64,3,1)
        self.conv3=nn.Conv2d(64,128,3,1)
        self.dropout1=nn.Dropout(0.25)
        self.dropout2=nn.Dropout(0.5)
        self.fc1=nn.Linear(6*6*128,1024)
        self.fc2=nn.Linear(1024,128)
        self.fc3=nn.Linear(128,10)
    
    def forward(self,x):
        x=self.conv1(x)
        x=F.relu(x)
        x=self.conv2(x)
        x=F.relu(x)
        x=F.max_pool2d(x,2)
        x=self.conv3(x)
        x=F.relu(x)
        x=F.max_pool2d(x,2)
        x=torch.flatten(x,1)
        x=self.fc1(x)
        x=F.relu(x)
        x=self.dropout2(x)
        x=self.fc2(x)
        x=F.relu(x)
        x=self.fc3(x)
        output=F.log_softmax(x,dim=1)
        return output

def train(model,device,train_loader,optimizer):
    model.train()
    for (data,target) in train_loader:
        data,target=data.to(device),target.to(device)
        optimizer.zero_grad()
        output=model(data)
        loss=F.nll_loss(output,target)
        loss.backward()
        optimizer.step()

def test(model,device,test_loader):
    model.eval()
    test_loss=0
    correct=0
    with torch.no_grad():
        for (data,target) in test_loader:
            data,target=data.to(device),target.to(device)
            output=model(data)
            test_loss+=F.nll_loss(output,target,reduction='sum').item()
            pred=output.argmax(dim=1,keepdim=True)
            correct+=pred.eq(target.view_as(pred)).sum().item()
    test_loss/=len(test_loader.dataset)
    print('测试结果：平均损失函数值：{:.4f},正确率：{}/{} ({:.0f}%)'.format(test_loss,correct,len(test_loader.dataset),100.*correct/len(test_loader.dataset)))

def main():
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914,0.4822,0.4465),(0.247,0.2435,0.2616)),
    ])
    train_kwargs={'batch_size':64}
    test_kwargs={'batch_size':1000}
    if torch.cuda.is_available():
        device=torch.device('cuda')
        num_cpus=os.cpu_count()
        cuda_kwargs={'num_workers':num_cpus,'pin_memory':True,'shuffle':True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)
    cifar10_train = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    cifar10_test=datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(cifar10_train, **train_kwargs)
    test_loader=torch.utils.data.DataLoader(cifar10_test, **test_kwargs)

    model=CNNClassifier().to(device)
    optimizer=optim.Adadelta(model.parameters(),lr=1.0)
    scheduler=StepLR(optimizer,step_size=1,gamma=0.7)

    epoches=10
    start=time.time()

    for epoch in range(1,epoches+1):
        train(model,device,train_loader,optimizer)
        print("批次：{}，".format(epoch),end='')
        test(model,device,test_loader)
        scheduler.step()
    
    end=time.time()
    print("训练时间：{:.2f}s".format(end-start))
    
    torch.save(model.state_dict(),'cifar10_cnn.pt')

if __name__=='__main__':
    main()