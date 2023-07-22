import argparse
import logging
import os
import numpy as np
import torch
import torch.nn as nn
from dataset import tensegrityDataset
from model import Ensemble_KF_low
from model import Ensemble_KF_no_action
from model import Forward_model_stable
from optimizer import build_optimizer
from optimizer import build_lr_scheduler
from einops import rearrange, repeat
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.utils.tensorboard import SummaryWriter
import math
import time
import pickle


class transform:
    def __init__(self):
        super(transform, self).__init__()
        parameters = pickle.load(
            open("./dataset/processed_data/parameter_52.pkl", "rb")
        )
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


class inference:
    def __init__(self):
        self.batch_size = 1
        self.dim_x = 7
        self.dim_z = 7
        self.dim_a = 40
        self.num_ensemble = 32
        self.mode = "test"
        self.model = Ensemble_KF_low(
            self.num_ensemble, self.dim_x, self.dim_z, self.dim_a
        )
        self.forward_model_ = Forward_model_stable(
            self.num_ensemble, self.dim_x, self.dim_a
        )
        # Check model type
        if not isinstance(self.model, nn.Module):
            raise TypeError("model must be an instance of nn.Module")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            self.model.cuda()
            self.forward_model_.cuda()

        # Load the pretrained model
        if torch.cuda.is_available():
            checkpoint = torch.load("continue-model-178800")
            self.model.load_state_dict(checkpoint["model"])
        else:
            checkpoint = torch.load(
                "continue-model-178800", map_location=torch.device("cpu")
            )
            self.model.load_state_dict(checkpoint["model"])
        self.model.eval()
        self.forward_model_.process_model.load_state_dict(
            self.model.process_model.state_dict()
        )

        self.transform_ = transform()
        self.utils_ = utils(self.num_ensemble, self.dim_x, self.dim_z)
        self.pe = self.utils_.positionalencoding1d(6, 20)

    def generate_mask(self, selection):
        """
        selection means the index of the row sensor to remove
        i.e. selection = [1] -> remove the 1st imu readings
             selection = [3,4] -> remove the 3td, and 4th imy readings
        """
        if len(selection) == 0:
            index = []
        else:
            index = []
            for i in range(len(selection)):
                if selection[i] == 1:
                    for j in range(6):
                        idx_1 = 0 + j
                        index.append(idx_1)
                if selection[i] == 2:
                    for j in range(6):
                        idx_1 = 6 + j
                        index.append(idx_1)
                if selection[i] == 3:
                    for j in range(6):
                        idx_1 = 12 + j
                        index.append(idx_1)
                if selection[i] == 4:
                    for j in range(6):
                        idx_1 = 18 + j
                        index.append(idx_1)
                if selection[i] == 5:
                    for j in range(6):
                        idx_1 = 24 + j
                        index.append(idx_1)
        tmp = np.ones((30, 128))
        if len(index) != 0:
            tmp[index, :] = tmp[index, :] * 0
        mask = torch.tensor(tmp, dtype=torch.float32)
        return mask

    def test(self):
        # Load recorded test data
        #######################################################################################
        ######################## for deployment we use real-time data ########################
        """
        NEED MODIFY!!!
        This is the entry point for ROS, I am now just using the recorded data
        make sure the data comes in the save format as the .pkl file
        once the data format is correct, the model is runable
        """
        data = pickle.load(open("./dataset/processed_data/train_dataset_16.pkl", "rb"))
        print(data["state_gt"].shape)
        print(data["obs"].shape)
        print(data["action"].shape)
        print(data["code"].shape)
        #######################################################################################
        #######################################################################################

        # save the result
        result = {}
        pre_save = []
        gt_save = []

        # for i in range (data['obs'].shape[0]):
        for i in range(1700):
            if i == 0:
                # initialize the filter
                state_init = np.zeros(7)
                state_init = torch.tensor(state_init, dtype=torch.float32)
                state_pre = rearrange(state_init, "(k dim) -> k dim", k=1)
                state_ensemble = self.utils_.format_state(state_pre)
                state_pre = rearrange(state_pre, "(bs k) dim -> bs k dim", bs=1)
                state_ensemble = rearrange(
                    state_ensemble, "(bs k) dim -> bs k dim", bs=1
                )
                state_ensemble = state_ensemble.to(self.device)
            elif i == 1:
                # initialize the filter
                state_init = state
                state_pre = rearrange(state_init, "bs k dim -> (bs k) dim", k=1)
                state_ensemble = self.utils_.format_state(state_pre)
                state_pre = rearrange(state_pre, "(bs k) dim -> bs k dim", bs=1)
                state_ensemble = rearrange(
                    state_ensemble, "(bs k) dim -> bs k dim", bs=1
                )
                state_ensemble = state_ensemble.to(self.device)

            ######################## for deployment we use real-time data ########################
            #######################################################################################
            """
            NEED MODIFY!!!
            This is the entry point for ROS, I am now just using the recorded data 
            for real-time daployment, pass ROS topic data and process the data
            and load the data as it is here
            """
            state_gt = torch.tensor(data["state_gt"][i], dtype=torch.float32)
            obs = torch.tensor(data["obs"][i], dtype=torch.float32)
            action = torch.tensor(data["action"][i], dtype=torch.float32)
            code = data["code"][i]
            #######################################################################################
            #######################################################################################

            state_gt = rearrange(state_gt, "(k dim) -> k dim", k=1)
            obs = rearrange(obs, "(k dim) -> k dim", k=1)
            action = rearrange(action, "(k dim) -> k dim", k=1)

            # apply the transformation
            obs = self.transform_.obs_transform(obs).to(dtype=torch.float32)
            action = self.transform_.action_transform(action).to(dtype=torch.float32)

            # reshape
            obs = rearrange(obs, "k dim -> (k dim)")
            # add pos embedding to the obs
            obs[0:6] = self.pe[int(code[0] - 1), :] + obs[0:6]
            obs[6:12] = self.pe[int(code[1] - 1), :] + obs[6:12]
            obs[12:18] = self.pe[int(code[2] - 1), :] + obs[12:18]
            obs[18:24] = self.pe[int(code[3] - 1), :] + obs[18:24]
            obs[24:30] = self.pe[int(code[4] - 1), :] + obs[24:30]
            obs = rearrange(obs, "(k dim) -> k dim", k=1)
            action = repeat(action, "k dim -> n k dim", n=self.num_ensemble)
            action = rearrange(action, "n k dim -> (n k) dim")

            # reshape to feed to the model
            obs = rearrange(obs, "(bs k) dim -> bs k dim", bs=1)
            action = rearrange(action, "(bs k) dim -> bs k dim", bs=1)
            state_gt = rearrange(state_gt, "(bs k) dim -> bs k dim", bs=1)
            state_gt = state_gt.to(self.device)
            obs = obs.to(self.device)
            action = action.to(self.device)

            selection = []  # -> try different combination by remove modalites
            mask = self.generate_mask(selection)
            mask = mask.to(self.device)

            with torch.no_grad():
                if i == 0:
                    ensemble = state_ensemble
                    state = state_pre
                    input_state = (ensemble, state)
                    obs_action = (action, obs)
                    output = self.model(obs_action, input_state, mask)
                    obs_p = output[3]  # -> learned observation
                    state = obs_p
                elif i == 1:
                    ensemble = state_ensemble
                    state = state
                    input_state = (ensemble, state)
                    obs_action = (action, obs)
                    output = self.model(obs_action, input_state, mask)
                    ensemble = output[0]  # -> ensemble estimation
                    state = output[1]  # -> final estimation
                    obs_p = output[3]  # -> learned observation
                else:
                    ensemble = ensemble
                    state = state
                    input_state = (ensemble, state)
                    if i >= 1500:
                        output = self.forward_model_(input_state, action)
                        ensemble = output[0]  # -> ensemble estimation
                        state = output[1]  # -> final estimation
                    else:
                        obs_action = (action, obs)
                        output = self.model(obs_action, input_state, mask)
                        ensemble = output[0]  # -> ensemble estimation
                        state = output[1]  # -> final estimation
                        obs_p = output[3]  # -> learned observation

                final_state = state
                final_state = rearrange(final_state, "bs k dim -> (bs k) dim", k=1)
                final_state = self.transform_.state_inv_transform(final_state)
                final_state = rearrange(final_state, "(bs k) dim -> bs k dim", bs=1)
                final_state = final_state.cpu().detach().numpy()
                state_gt = state_gt.cpu().detach().numpy()

                # print('============')
                # print(final_state)
                # print(state_gt)
                pre_save.append(final_state)
                gt_save.append(state_gt)
        result["state"] = pre_save
        result["gt"] = gt_save
        with open("result_test.pkl", "wb") as f:
            pickle.dump(result, f)


def main():
    inference_ = inference()
    inference_.test()


if __name__ == "__main__":
    main()
