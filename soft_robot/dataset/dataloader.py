import os
import random
from PIL import Image
import cv2
import numpy as np
from torch.utils.data import Dataset
import pickle
import torch
from einops import rearrange, repeat
from torch.distributions.multivariate_normal import MultivariateNormal
import math
import pdb


class transform:
    def __init__(self, args):
        super(transform, self).__init__()
        self.args = args
        parameters = pickle.load(open(self.args.mode.parameter_path, "rb"))
        self.state_m = parameters["state_m"]
        self.state_std = parameters["state_std"]
        self.obs_m = parameters["obs_m"]
        self.obs_std = parameters["obs_std"]
        self.action_m = parameters["action_m"]
        self.action_std = parameters["action_std"]

    def state_transform(self, state):
        """
        state -> [num_ensemble, dim_x]
        """
        state = (state - self.state_m) / self.state_std
        return state

    def obs_transform(self, obs):
        """
        obs -> [num_ensemble, dim_z]
        """
        obs = (obs - self.obs_m) / self.obs_std
        return obs

    def action_transform(self, action):
        """
        action -> [num_ensemble, dim_a]
        """
        action = (action - self.action_m) / self.action_std
        return action

    def state_inv_transform(self, state):
        """
        state -> [num_ensemble, dim_x]
        """
        state = (state * self.state_std) + self.state_m
        return state

    def obs_inv_transform(self, obs):
        """
        obs -> [num_ensemble, dim_z]
        """
        obs = (obs * self.obs_std) + self.obs_m
        return obs

    def action_inv_transform(self, action):
        """
        action -> [num_ensemble, dim_a]
        """
        action = (action * self.action_std) + self.action_m
        return action


class utils:
    def __init__(self, num_ensemble, dim_x, dim_z):
        self.num_ensemble = num_ensemble
        self.dim_x = dim_x
        self.dim_z = dim_z

    def multivariate_normal_sampler(self, mean, cov, k):
        sampler = MultivariateNormal(mean, cov)
        return sampler.sample((k,))

    def format_state(self, state):
        state = repeat(state, "k dim -> n k dim", n=self.num_ensemble)
        state = rearrange(state, "n k dim -> (n k) dim")
        cov = torch.eye(self.dim_x) * 0.05
        init_dist = self.multivariate_normal_sampler(
            torch.zeros(self.dim_x), cov, self.num_ensemble
        )
        state = state + init_dist
        state = state.to(dtype=torch.float32)
        return state

    def positionalencoding1d(self, d_model, length):
        """
        :param d_model: dimension of the model
        :param length: length of positions
        :return: length*d_model position matrix
        """
        if d_model % 2 != 0:
            raise ValueError(
                "Cannot use sin/cos positional encoding with "
                "odd dim (got dim={:d})".format(d_model)
            )
        pe = torch.zeros(length, d_model)
        position = torch.arange(0, length).unsqueeze(1)
        div_term = torch.exp(
            (
                torch.arange(0, d_model, 2, dtype=torch.float)
                * -(math.log(10000.0) / d_model)
            )
        )
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)

        return pe


class tensegrityDataset(Dataset):
    # Basic Instantiation
    def __init__(self, args, mode):
        self.args = args
        self.mode = mode
        if self.mode == "train":
            self.dataset_path = self.args.train.data_path
            self.num_ensemble = self.args.train.num_ensemble
        elif self.mode == "test":
            self.dataset_path = self.args.test.data_path
            self.num_ensemble = self.args.test.num_ensemble
        self.dataset = pickle.load(open(self.dataset_path, "rb"))
        self.dataset_length = len(self.dataset["state_gt"])
        self.dim_x = self.args.train.dim_x
        self.dim_z = self.args.train.dim_z
        self.dim_a = self.args.train.dim_a
        self.transform_ = transform(self.args)
        self.utils_ = utils(
            self.num_ensemble, self.args.train.dim_x, self.args.train.dim_z
        )
        self.pe = self.utils_.positionalencoding1d(6, 20)

    # Length of the Dataset
    def __len__(self):
        # self.dataset_length = 50
        return self.dataset_length - 2

    # Fetch an item from the Datasetcd
    def __getitem__(self, idx):
        state_gt = torch.tensor(self.dataset["state_gt"][idx], dtype=torch.float32)
        state_pre = torch.tensor(self.dataset["state_pre"][idx], dtype=torch.float32)
        obs = torch.tensor(self.dataset["obs"][idx], dtype=torch.float32)
        action = torch.tensor(self.dataset["action"][idx], dtype=torch.float32)
        code = self.dataset["code"][idx]
        sample_freq = torch.tensor(
            self.dataset["sample_freq"][idx], dtype=torch.float32
        )

        state_gt = rearrange(state_gt, "(k dim) -> k dim", k=1)
        state_pre = rearrange(state_pre, "(k dim) -> k dim", k=1)
        obs = rearrange(obs, "(k dim) -> k dim", k=1)
        action = rearrange(action, "(k dim) -> k dim", k=1)

        # apply the transformation
        state_gt = self.transform_.state_transform(state_gt).to(dtype=torch.float32)
        state_pre = self.transform_.state_transform(state_pre).to(dtype=torch.float32)
        obs = self.transform_.obs_transform(obs).to(dtype=torch.float32)

        obs = rearrange(obs, "k dim -> (k dim)")
        # add pos embedding to the obs
        obs[0:6] = self.pe[int(code[0] - 1), :] + obs[0:6]
        obs[6:12] = self.pe[int(code[1] - 1), :] + obs[6:12]
        obs[12:18] = self.pe[int(code[2] - 1), :] + obs[12:18]
        obs[18:24] = self.pe[int(code[3] - 1), :] + obs[18:24]
        obs[24:30] = self.pe[int(code[4] - 1), :] + obs[24:30]

        obs = rearrange(obs, "(k dim) -> k dim", k=1)
        action = self.transform_.action_transform(action).to(dtype=torch.float32)
        action = repeat(action, "k dim -> n k dim", n=self.num_ensemble)
        action = rearrange(action, "n k dim -> (n k) dim")
        state_ensemble = self.utils_.format_state(state_pre)
        sample_freq = repeat(sample_freq, "dim -> n dim", n=self.num_ensemble)

        return state_gt, state_pre, obs, action, state_ensemble, sample_freq


############ only for testing ############
# if __name__ == '__main__':
#     dataset = tensegrityDataset(32,5,2, 'train')
#     dataloader = torch.utils.data.DataLoader(dataset, batch_size=2,
#                                           shuffle=True, num_workers=1)
#     for state_gt, state_pre, obs, raw_obs, state_ensemble in dataloader:
#         print(state_ensemble.shape)
#         print("check -------- ",state_ensemble.dtype)
#         print(raw_obs.shape)
