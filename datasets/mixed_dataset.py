import os
import os.path as osp
from glob import glob
import numpy as np
import cv2
import h5py



# My libraries
from .cleargrasp import ClearGraspSynthetic
from .omniverse_object import OmniverseObject




class MixedDataset:
    """
    Mixed Dataset for MindSpore combining ClearGraspSynthetic and OmniverseObject datasets.
    """
    def __init__(self, cleargrasp_root_dir, omniverse_root_dir, split='train', **kwargs):
        """
        Initialization.

        Parameters
        ----------
        cleargrasp_root_dir: str, the root directory of the ClearGraspSynthetic dataset.
        omniverse_root_dir: str, the root directory of the OmniverseObject dataset.
        split: str, the dataset split option.
        """
        self.cleargrasp_syn_dataset = ClearGraspSynthetic(cleargrasp_root_dir, split, **kwargs)
        self.omniverse_dataset = OmniverseObject(omniverse_root_dir, split, **kwargs)
        self.cleargrasp_syn_len = len(self.cleargrasp_syn_dataset)
        self.omniverse_len = len(self.omniverse_dataset)

    def __getitem__(self, index):
        """
        Get a single item from the dataset.

        Parameters
        ----------
        index: int, the index of the item.

        Returns
        -------
        A single data item.
        """
        if index < self.cleargrasp_syn_len:
            return self.cleargrasp_syn_dataset[index]
        else:
            return self.omniverse_dataset[index - self.cleargrasp_syn_len]

    def __len__(self):
        """
        Get the total number of items in the dataset.

        Returns
        -------
        int, the total number of items.
        """
        return self.cleargrasp_syn_len + self.omniverse_len