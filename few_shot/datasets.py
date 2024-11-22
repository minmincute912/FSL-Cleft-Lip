from torch.utils.data import Dataset
import torch
from PIL import Image
from torchvision import transforms
from skimage import io
from tqdm import tqdm
import pandas as pd
import numpy as np
import os

from config import DATA_PATH

class CleftLipDataset(Dataset):
    def __init__(self, subset):
        """
        Dataset class representing Cleft Lip dataset

        # Arguments:
        subset: Whether the dataset represents the background or evaluation set
        """
        if subset not in ('background','evaluation'):
            raise ValueError('subset must be one of (background, evaluation)')
        self.subset = subset
        self.df = pd.DataFrame(self.index_subset(self.subset))
        # Index of dataframe has direct correspondence to item in dataset
        self.df = self.df.assign(id=self.df.index.values)

        # Convert arbitrary class names of dataset to ordered 0-(num_speakers - 1) integers
        self.unique_characters = sorted(self.df['class_name'].unique())
        self.class_name_to_id = {self.unique_characters[i]: i for i in range(self.num_classes())}
        self.df = self.df.assign(class_id=self.df['class_name'].apply(lambda c: self.class_name_to_id[c]))

        # Create dicts
        self.datasetid_to_filepath = self.df.to_dict()['filepath']
        self.datasetid_to_class_id = self.df.to_dict()['class_id']

        # Setup transforms
        self.transform = transforms.Compose(
            [
                transforms.Resize((224,224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])
            ]
        )
    def __getitem__(self,item):
        instance = Image.open(self.datasetid_to_filepath[item]).convert('RGB')
        instance = self.transform(instance)
        label = self.datasetid_to_class_id[item]
        return instance, label
    def __len__(self):
        return len(self.df)
    def num_classes(self):
        return len(self.df['class_name'].unique())

    @staticmethod
    def index_subset(subset):
        images = []
        subset_path = os.path.join(DATA_PATH, f'images_{subset}')
        print('Indexing {} ...'.format(subset))
        # Quick first pass to find total for tqdm bar
        for group_name in os.listdir(subset_path):
            group_path = os.path.join(subset_path, group_name)
            if not os.path.isdir(group_path):
                continue
            for img_name in tqdm(os.listdir(group_path)):
                if img_name.endswith(('png')):
                    images.append({'subset':subset,
                    'class_name':group_name,
                    'filepath': os.path.join(group_path,img_name)})
        return images