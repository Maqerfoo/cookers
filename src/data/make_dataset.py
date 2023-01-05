# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

import os
import wget
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader


class MyDataset(Dataset):
    def __init__(self, filepaths, transform=None):
        self.imgs = np.concatenate([np.load(f,allow_pickle=True)["images"] for f in filepaths])
        self.labels = np.concatenate([np.load(f,allow_pickle=True)["labels"] for f in filepaths])
        self.imgs = self.imgs.astype("float32").reshape(-1, 1, 28,28)
        self.labels = self.labels.astype("float32")
        self.transform = transform

    def __len__(self):
        return self.imgs.shape[0]
    def __getitem__(self, idx):
        if self.transform:
            self.imgs[idx] = self.transform(self.imgs[idx])
        return (self.imgs[idx], self.labels[idx])

@click.command()
@click.option('--input_filepath', default="data/raw", type=click.Path(exists=True), help="input filepath")
@click.option('--output_filepath', default="data/processed", type=click.Path(), help="output filepath")
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')
    
    #download dataset
    #for i in range(5):
    #    url = "https://github.com/SkafteNicki/dtu_mlops/blob/main/data/corruptmnist/train_{}.npz".format(i)
    #    wget.download(url, out=input_filepath)
    #wget.download("https://github.com/SkafteNicki/dtu_mlops/blob/main/data/corruptmnist/test.npz",out=input_filepath)

    transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0,), (1,))])
    train_files = []
    test_files = []
    for file in os.listdir(input_filepath):
        if file.startswith("train"):
            train_files.append(file)
        elif file.startswith("test"):
            test_files.append(file)

    train_files = [os.path.join(input_filepath,f) for f in train_files]
    test_files = [os.path.join(input_filepath,f) for f in test_files]
    train_dl = DataLoader(MyDataset(train_files,transform=transform), batch_size = 16)
    test_dl = DataLoader(MyDataset(test_files,transform=transform), batch_size = 16)

    torch.save(train_dl, os.path.join(output_filepath,"train_dataloader"))
    torch.save(test_dl, os.path.join(output_filepath,"test_dataloader"))

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())    
    main()
