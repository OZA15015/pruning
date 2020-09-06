import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import OrderedDict


# ToTensor：画像のグレースケール化（RGBの0~255を0~1の範囲に正規化）、Normalize：Z値化（RGBの平均と標準偏差を0.5で決め打ちして正規化）
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# トレーニングデータをダウンロード
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)

# テストデータをダウンロード
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=True, num_workers=2)




class Simplenet(nn.Module):
    def __init__(self):
        super(Simplenet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        #x = nn.Threshold(0.2, 0.0)#ActivationZeroThreshold(x)
        x = self.fc3(x)
        return x

def simplenet_cifar():
    model = Simplenet()
    return model


def test_accuracy(model):
    correct = 0
    total = 0
    # 勾配を記憶せず（学習せずに）に計算を行う
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))


def fix_model_state_dict(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k
        if name.startswith('module.'):
            name = name[7:]  # remove 'module.' of dataparallel
        new_state_dict[name] = v
    return new_state_dict

def main():
    model = simplenet_cifar()
    #model.cuda()
    #checkpoint = torch.load("simplenet_cifar/best.pth.tar")['state_dict']
    checkpoint = torch.load("logs/simple_net_pruner0819/best.pth.tar")['state_dict']
    new_state_dict = OrderedDict()
    for k, v in checkpoint.items():
        print(k)
        if 'module' in k:
            k = k.replace('module.', '')
        new_state_dict[k] = v

    #checkpoint = fix_model_state_dict(checkpoint)
    model.load_state_dict(new_state_dict)
    model.eval()
    test_accuracy(model)

if __name__ == "__main__":
    main()


