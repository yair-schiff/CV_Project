from __future__ import print_function

import argparse
import json
import os

import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

import pdb


########################################################################################################################
# Dataset
def ddsm_crop(image, target_dims):
    image = np.asarray(image)
    h, w = image.shape
    y = h // 2
    x = 0
    normalized_image = (image - image.mean(axis=(-2, -1), keepdims=1)) / image.std(axis=(-2, -1), keepdims=1)
    cropped_image = normalized_image[y - target_dims[0] // 2:y + target_dims[0] // 2, x:x + target_dims[1]]
    return Image.fromarray(cropped_image)


CROP_SIZE = (1500, 900)
default_transform = transforms.Compose([transforms.Resize((2048, 1024)),
                                        transforms.Lambda(lambda img: ddsm_crop(img, CROP_SIZE)),
                                        transforms.ToTensor()])


def create_classes():
    class_to_idx = {"normal": 0, "benign": 1, "malignant": 2}
    return class_to_idx


def get_classification(case_name):
    if "normal" in case_name:
        return "normal"
    elif "benign" in case_name:
        return "benign"
    else:
        return "malignant"


def get_label(name, image_anns):
    for image in image_anns:
        if image["file_name"] == name:
            label = get_classification(image["case_name"])
            return label


def get_brightened(name, image_anns):
    for image in image_anns:
        if image["file_name"] == name:
            return image["brightened"]


def make_dataset(data_dir, dataset, class_to_idx, exclude_brightened=False):
    with open(os.path.join(data_dir, "annotations/instances_{}.json".format(dataset)), "r") as annotations:
        ann_json = json.load(annotations)
        image_anns = ann_json["images"]
    images = []
    imgs_dir = os.path.join(data_dir, dataset)
    for root, _, fnames in sorted(os.walk(imgs_dir)):
        for fname in sorted(fnames):
            if "jpg" not in fname:
                continue
            path = os.path.join(root, fname)
            if exclude_brightened and get_brightened(fname, image_anns):
                continue
            label = get_label(fname, image_anns)
            item = (path, class_to_idx[label])
            images.append(item)
    return images


def greyscale_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    # with open(path, "rb") as f:
    img = Image.open(open(path, "rb"))
    return img


class DDSMDataset(torch.utils.data.Dataset):
    def __init__(self, root, dataset="train", transform=default_transform,
                 target_transform=None, loader=greyscale_loader, exclude_brightened=False):
        class_to_idx = create_classes()
        samples = make_dataset(root, dataset, class_to_idx, exclude_brightened)

        self.root = root
        self.loader = loader

        self.class_to_idx = class_to_idx
        self.samples = samples

        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


########################################################################################################################
# Models
class MyResNet(nn.Module):
    def __init__(self, desired_resnet, num_classes, only_train_heads=False):
        super(MyResNet, self).__init__()
        resnet_dict = {
            "resnet18": models.resnet18,
            "resnet34": models.resnet34,
            "resnet50": models.resnet50,
            "resnet152": models.resnet152
        }
        self.model = resnet_dict[desired_resnet](pretrained=True)
        num_ftrs = 461824  # self.model.fc.in_features
        if only_train_heads:
            for param in self.model.parameters():
                param.requires_grad = False
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), bias=False)
        self.model.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        output = self.model(x)
        return output


########################################################################################################################
# Training & Validation
def train(model, train_loader, optimizer, device, epoch, log_interval):
    model.train()
    print("Training epoch {}...".format(epoch))
    batch_idx = 0
    for (data, target) in train_loader:
        batch_idx += 1
        data, target = Variable(data).to(device), Variable(target).to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def validation(model, val_loader, device):
    model.eval()
    validation_loss = 0
    correct = 0
    for data, target in val_loader:
        with torch.no_grad():
            data, target = Variable(data).to(device), Variable(target).to(device)
        output = model(data)
        validation_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    validation_loss /= len(val_loader.dataset)
    print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        validation_loss, correct, len(val_loader.dataset),
        100. * correct / len(val_loader.dataset)))
    return correct


def main():
    parser = argparse.ArgumentParser(description='PyTorch DDSM Classification')
    parser.add_argument('--data', type=str, default='data', metavar='D',
                        help="folder where data is located.")
    parser.add_argument('--model-results', type=str, default='model_results', metavar='M',
                        help="folder where model results will be saved. Defaults to 'model_results/")
    parser.add_argument('--batch-size', type=int, default=2, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status (default: 10)')
    parser.add_argument('--checkpoint', type=str, default="", metavar='C',
                        help="Provide checkpoint model from which to load weights.")

    # Parse arguments
    args = parser.parse_args()
    data_dir = args.data
    model_res_dir = args.model_results
    batch_size = args.batch_size
    epochs = args.epochs
    lr = args.lr
    checkpoint = args.checkpoint
    train_heads = checkpoint == ""
    log_interval = args.log_interval
    torch.manual_seed(args.seed)

    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device != torch.device("cpu"):
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    print("Device: {}".format(device))

    # Load data
    train_loader = torch.utils.data.DataLoader(DDSMDataset(data_dir, dataset="train", exclude_brightened=True),
                                               batch_size=batch_size, shuffle=True, num_workers=1)
    val_loader = torch.utils.data.DataLoader(DDSMDataset(data_dir, dataset="val", exclude_brightened=True),
                                             batch_size=batch_size, shuffle=True, num_workers=1)
    # Load model
    model = MyResNet("resnet18", 3, only_train_heads=train_heads)
    if checkpoint != "":
        state_dict = torch.load(checkpoint) if torch.cuda.is_available() else torch.load(checkpoint, map_location='cpu')
        model.load_state_dict(state_dict)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Run training and validation
    if not os.path.isdir(model_res_dir):
        print(model_res_dir + " not found: making directory for results")
        os.mkdir(model_res_dir)
    for epoch in range(1, epochs + 1):
        train(model, train_loader, optimizer, device, epoch, log_interval)
        correct = validation(model, val_loader, device)
        model_file = os.path.join(model_res_dir, "model_stage" +
                                  str(1 if train_heads else 2) + "_" + str(epoch) + ".pth")
        torch.save(model.state_dict(), model_file)
        print("\nSaved model to " + model_file + ".")


if __name__ == "__main__":
    main()
