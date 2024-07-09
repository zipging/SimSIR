import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import ConcatDataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

import random
from tqdm import tqdm
from PIL import ImageFilter

normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


class GaussianBlur:
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x
    
class CombinedLoader:
    def __init__(self, loader1, loader2):
        self.loader1 = loader1
        self.loader2 = loader2
        self.iterator1 = iter(loader1)
        self.iterator2 = iter(loader2)
        self.batch_index = 0

    def __iter__(self):
        self.iterator1 = iter(self.loader1)
        self.iterator2 = iter(self.loader2)
        self.batch_index = 0
        return self

    def __next__(self):
        if self.iterator1 and self.iterator2:
            (data1, labels1) = next(self.iterator1)  # 从第一个 loader 获取数据和标签
            (data2, labels2) = next(self.iterator2)  # 从第二个 loader 获取数据和标签
            batch_size = data1.size(0)  # 基于第一个数据集的数据部分确定批次大小
            current_indices = list(range(self.batch_index, self.batch_index + batch_size))
            self.batch_index += batch_size
            return (data1, labels1), (data2, labels2), current_indices
        else:
            raise StopIteration

    
def get_cifar10(batch_size=128):
    augmentation = [
            transforms.RandomCrop(32, padding=4),
            #transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
            transforms.RandomApply(
                [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8  # not strengthened
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]


    train_dataset_q = datasets.CIFAR10(root='data', train=True, download=True, transform=transforms.Compose(augmentation))
    train_dataset_k = datasets.CIFAR10(root='data', train=True, download=True, transform=transforms.Compose(augmentation))
    train_loader_q = torch.utils.data.DataLoader(dataset=train_dataset_q, batch_size=batch_size)
    train_loader_k = torch.utils.data.DataLoader(dataset=train_dataset_k, batch_size=batch_size)
    train_loader = CombinedLoader(train_loader_q, train_loader_k)
    
    return train_loader
