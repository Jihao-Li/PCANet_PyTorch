import torch
import argparse
import numpy as np
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10


class Cutout(object):
    """ cutout operation """
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


def _data_transforms_cifar10(args):
    """
    preprocess cifar10 dataset
    :param args: hyper parameter
    :return: 
    """
    CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
    CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

    # trainset init operation
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

    if args.cutout:
        train_transform.transforms.append(Cutout(args.cutout_length))

    # vaildset init operation
    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

    return train_transform, valid_transform


def load_cifar10(args):
    """
    load cifar10 dataset
    :param args: hyper parameter
    :return: 
    """
    # data preprocessing
    train_transform, valid_transform = _data_transforms_cifar10(args)

    # load data
    train_data = CIFAR10(root=args.dataset_path, train=True, download=True, transform=train_transform)
    test_data = CIFAR10(root=args.dataset_path, train=False, download=True, transform=valid_transform)

    # set up data queue
    num_train = len(train_data)           # number of train set
    indices = list(range(num_train))
    split = int(np.floor(args.train_portion * num_train))

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
        pin_memory=True, num_workers=2)
    valid_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
        pin_memory=True, num_workers=2)
    test_queue = torch.utils.data.DataLoader(
        test_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=2)

    return train_queue, valid_queue, test_queue


if __name__ == "__main__":
    parser = argparse.ArgumentParser("mnist_dataset")
    parser.add_argument('--dataset_path', type=str, default='/dataset/cifar10', help='location of the data corpus')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
    parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
    args = parser.parse_args()

    train_queue, valid_queue, test_queue = load_cifar10(args)
    for step, (train_images, train_labels) in enumerate(train_queue):
        print("step: ", step)
        # print(train_images, train_labels)
        continue
    print("step:", step)
