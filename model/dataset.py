import torch
import cv2
import numpy as np


class BEVImageDataset(torch.utils.data.Dataset):
    def __init__(self, input_filepaths, target_filepaths, map_filepaths=None):
        self.input_filepaths = input_filepaths
        self.target_filepaths = target_filepaths
        self.map_filepaths = map_filepaths

        if map_filepaths is not None:
            assert len(input_filepaths) == len(map_filepaths)

        assert len(input_filepaths) == len(target_filepaths)

    def __len__(self):
        return len(self.input_filepaths)

    def __getitem__(self, idx):
        input_filepath = self.input_filepaths[idx]
        target_filepath = self.target_filepaths[idx]

        sample_token = input_filepath.split("/")[-1].replace("_input.png", "")

        im = cv2.imread(input_filepath, cv2.IMREAD_UNCHANGED)

        if self.map_filepaths:
            map_filepath = self.map_filepaths[idx]
            map_im = cv2.imread(map_filepath, cv2.IMREAD_UNCHANGED)
            im = np.concatenate((im, map_im), axis=2)

        target = cv2.imread(target_filepath, cv2.IMREAD_UNCHANGED)

        im = im.astype(np.float32) / 255
        target = target.astype(np.int64)

        im = torch.from_numpy(im.transpose(2, 0, 1))
        target = torch.from_numpy(target)

        return im, target, sample_token
