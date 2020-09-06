import sys    
sys.path.append('/home/oza/pre-experiment/speeding/testFB/FBNet')

import torch                                                                                                                            
from torch import nn
import numpy as np
from tensorboardX import SummaryWriter
import argparse
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms

#import distiller.models.cifar10.fbnet_building_blocks.fbnet_builder as fbnet_builder
import fbnet_building_blocks.fbnet_builder as fbnet_builder
from collections import OrderedDict


# ToTensor：画像のグレースケール化（RGBの0~255を0~1の範囲に正規化）、Normalize：Z値化（RGBの平均と標準偏差を0.5で決め打ちして正規化）
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
 
# トレーニングデータをダウンロード
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)                                                                    
 
# テストデータをダウンロード
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=10000, shuffle=True, num_workers=16)
    
manual_seed = 1 
np.random.seed(manual_seed)
torch.manual_seed(manual_seed)
torch.cuda.manual_seed_all(manual_seed)
torch.backends.cudnn.benchmark = True

def fbneta_cifar():
    model = fbnet_builder.get_model('fbnet_a', cnt_classes=10).cuda()
    return model

def test_accuracy(model):
    correct = 0
    total = 0
    # 勾配を記憶せず（学習せずに）に計算を行う
    #with torch.no_grad():
    for data in testloader:
        images, labels = data
        images = images.to('cuda')
        labels = labels.to('cuda')
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))


def main():
    model = fbneta_cifar()
    #checkpoint = torch.load("/home/oza/pre-experiment/speeding/FBNet/architecture_functions/logs/best_model.pth", map_location="cuda")
    #checkpoint = torch.load("/home/oza/pre-experiment/speeding/distiller/examples/classifier_compression/logs/test_FBnetA0826/best.pth.tar")['state_dict']
    #checkpoint = torch.load("/home/oza/pre-experiment/speeding/test_dist/logs/test_FBnetA0826/best.pth.tar")['state_dict']
    #checkpoint = torch.load("/home/oza/pre-experiment/speeding/testFB/distiller/examples/classifier_compression/logs/fbnet_a_0828_2/best.pth.tar")['state_dict']
    checkpoint = torch.load("/home/oza/pre-experiment/speeding/testFB/distiller/examples/classifier_compression/logs/2020.08.29-035310/best.pth.tar", map_location="cuda")['state_dict']
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

if __name__== "__main__":
    main()
