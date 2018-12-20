from __future__ import print_function

import argparse
import os

import pandas as pd
import torch
from torch.autograd import Variable

# Internal dependencies
from classification_tumor import MyResNet
from classification_tumor import create_classes, default_transform, greyscale_loader


########################################################################################################################
# Dataset - INbreast
def make_dataset(data_dir, dataset, class_to_idx, cases):
    items = []
    imgs_dir = os.path.join(data_dir, dataset)
    df_masks = pd.DataFrame.from_csv(os.path.join(cases, "INbreast_mask.csv"))
    df_files_to_ids = pd.DataFrame.from_csv(os.path.join(cases, "INbreast_file_to_id.csv"))
    mask_dict = dict(zip(list(df_masks["File Name"]), list(df_masks["Mask"])))
    files_dict = dict(zip(list(df_files_to_ids["image_ids"])), list(df_files_to_ids["file_ids"]))
    for img_id, file_id in files_dict.items():
        item = (os.path.join(imgs_dir, img_id), mask_dict[file_id])
        items.append(item)
    return items


class INbreast(torch.utils.data.Dataset):
    def __init__(self, root, cases, dataset="test", transform=default_transform,
                 target_transform=None, loader=greyscale_loader):
        class_to_idx = create_classes()
        samples = make_dataset(root, dataset, class_to_idx, cases)

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
        fmt_str += '    Number of data points: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


########################################################################################################################
# Evaluation
def evaluate(model, test_loader, device):
    model.eval()
    image_idx = 0
    for data, target in test_loader:
        image_idx += 1
        with torch.no_grad():
            data, target = Variable(data).to(device), Variable(target).to(device)
        output = model(data)
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        right_or_wrong = "correct" if pred == output else "wrong"
        print("Image {}: True label = {}; Predicted Label = {}. Got this image {}".format(image_idx, target, pred,
                                                                                          right_or_wrong))


def main():
    parser = argparse.ArgumentParser(description='PyTorch INbreast Evaluation')
    parser.add_argument('--cases', type=str, default='INbreast/cases', metavar='D',
                        help="folder where INbreast cases are located.")
    parser.add_argument('--data', type=str, default='INbreast/data', metavar='D',
                        help="folder where INbreast processed data are located.")
    parser.add_argument('--model', type=str, default='model_results', metavar='M',
                        help="path to model to be used in evaluation")
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')

    # Parse arguments
    args = parser.parse_args()
    cases_dir = args.cases
    data_dir = args.data
    model_path = args.model
    torch.manual_seed(args.seed)

    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device != torch.device("cpu"):
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    print("Device: {}".format(device))

    # Load data
    test_loader = torch.utils.data.DataLoader(INbreast(data_dir, cases_dir, dataset="test", exclude_brightened=True),
                                              batch_size=1, shuffle=False, num_workers=1)
    # Load model
    model = MyResNet("resnet18", 2, only_train_heads=False, pretrained=False)
    state_dict = torch.load(model_path) if torch.cuda.is_available() else torch.load(model_path, map_location='cpu')
    model.load_state_dict(state_dict)
    model = model.to(device)

    # Run evaluation:
    evaluate(model, test_loader, device)


if __name__ == "__main__":
    main()
