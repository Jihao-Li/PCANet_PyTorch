import torch
import argparse
import numpy as np
import torchvision.transforms as transforms
from torchvision.datasets import mnist


def _data_transforms_mnist():
    """
    preprocess mnist dataset
    :return: 
    """
    # trainset init operation
    train_transform = transforms.Compose([
        transforms.RandomCrop(28, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    # vaildset init operation
    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    return train_transform, valid_transform


def load_train_mnist(args):
    """
    load mnist dataset for training
    :param args: hyper parameter
    :return: 
    """
    # data preprocessing
    train_transform, _ = _data_transforms_mnist()

    # load data
    train_data = mnist.MNIST(root=args.dataset_path, train=True, download=True, transform=train_transform)

    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(args.train_portion * num_train))

    # set up data queue
    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
        pin_memory=True, num_workers=2)
    valid_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
        pin_memory=True, num_workers=2)

    return train_queue, valid_queue


def load_test_mnist(args):
    """
    load mnist dataset for testing
    :param args: hyper parameter
    :return: 
    """
    # data preprocessing
    _, valid_transform = _data_transforms_mnist()

    # load data
    test_data = mnist.MNIST(root=args.dataset_path, train=False, download=True, transform=valid_transform)
    num_test = len(test_data)

    # set up data queue
    test_queue = torch.utils.data.DataLoader(
        test_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=2)

    return test_queue, num_test


def load_mnist(args):
    """
    load mnist dataset
    :param args: hyper parameter
    :return: 
    """
    # data preprocessing
    train_transform, valid_transform = _data_transforms_mnist()

    # load data
    train_data = mnist.MNIST(root=args.dataset_path, train=True, download=True, transform=train_transform)
    test_data = mnist.MNIST(root=args.dataset_path, train=False, download=True, transform=valid_transform)

    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(args.train_portion * num_train))

    # set up data queue
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
    parser.add_argument('--dataset_path', type=str, default='/dataset/', help='location of the data corpus')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--train_portion', type=float, default=0.7, help='portion of training data')
    args = parser.parse_args()

    train_queue, valid_queue, test_queue = load_mnist(args)
    for step, (train_images, train_labels) in enumerate(train_queue):
        # print("step: ", step)
        # print(train_images, train_labels)
        continue
    print("step:", step)
