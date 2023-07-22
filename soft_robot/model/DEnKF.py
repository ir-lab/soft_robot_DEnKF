from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
from bayesian_torch.layers.flipout_layers.linear_flipout import LinearFlipout
import torchvision.models as models
from einops import rearrange, repeat
import numpy as np
import math
import pdb


class ProcessModel(nn.Module):
    """
    process model takes a state or a stack of states (t-n:t-1) and
    predict the next state t. the process model is flexiable, we can inject the known
    dynamics into it, we can also change the model architecture which takes sequential
    data as input

    input -> [batch_size, num_ensemble, dim_x]
    output ->  [batch_size, num_ensemble, dim_x]
    """

    def __init__(self, num_ensemble, dim_x):
        super(ProcessModel, self).__init__()
        self.num_ensemble = num_ensemble
        self.dim_x = dim_x

        self.bayes1 = LinearFlipout(in_features=self.dim_x, out_features=64)
        self.bayes2 = LinearFlipout(in_features=64, out_features=512)
        self.bayes3 = LinearFlipout(in_features=512, out_features=256)
        self.bayes4 = LinearFlipout(in_features=256, out_features=self.dim_x)

    def forward(self, last_state):
        batch_size = last_state.shape[0]
        last_state = rearrange(
            last_state, "bs k dim -> (bs k) dim", bs=batch_size, k=self.num_ensemble
        )
        x, _ = self.bayes1(last_state)
        x = F.relu(x)
        x, _ = self.bayes2(x)
        x = F.relu(x)
        x, _ = self.bayes3(x)
        x = F.relu(x)
        update, _ = self.bayes4(x)
        state = last_state + update
        state = rearrange(
            state, "(bs k) dim -> bs k dim", bs=batch_size, k=self.num_ensemble
        )
        return state


class ProcessModelAction(nn.Module):
    """
    process model takes a state or a stack of states (t-n:t-1) and
    predict the next state t. this process model takes in the state and actions
    and outputs a predicted state

    input -> [batch_size, num_ensemble, dim_x]
    action -> [batch_size, num_ensemble, dim_a]
    output ->  [batch_size, num_ensemble, dim_x]
    """

    def __init__(self, num_ensemble, dim_x, dim_a):
        super(ProcessModelAction, self).__init__()
        self.num_ensemble = num_ensemble
        self.dim_x = dim_x
        self.dim_a = dim_a

        # channel for state variables
        self.bayes1 = LinearFlipout(in_features=self.dim_x, out_features=64)
        self.bayes2 = LinearFlipout(in_features=64, out_features=128)
        self.bayes3 = LinearFlipout(in_features=128, out_features=64)

        # channel for action variables
        self.bayes_a1 = LinearFlipout(in_features=self.dim_a, out_features=64)
        self.bayes_a2 = LinearFlipout(in_features=64, out_features=128)
        self.bayes_a3 = LinearFlipout(in_features=128, out_features=64)

        # merge them
        self.bayes4 = LinearFlipout(in_features=128, out_features=64)
        self.bayes5 = LinearFlipout(in_features=64, out_features=self.dim_x)

    def forward(self, last_state, action):
        batch_size = last_state.shape[0]
        last_state = rearrange(
            last_state, "bs k dim -> (bs k) dim", bs=batch_size, k=self.num_ensemble
        )
        action = rearrange(
            action, "bs k dim -> (bs k) dim", bs=batch_size, k=self.num_ensemble
        )

        # branch for the state variables
        x, _ = self.bayes1(last_state)
        x = F.relu(x)
        x, _ = self.bayes2(x)
        x = F.relu(x)
        x, _ = self.bayes3(x)
        x = F.relu(x)

        # branch for the action variables
        y, _ = self.bayes_a1(action)
        y = F.relu(y)
        y, _ = self.bayes_a2(y)
        y = F.relu(y)
        y, _ = self.bayes_a3(y)
        y = F.relu(y)

        # merge branch
        merge = torch.cat((x, y), axis=1)
        merge, _ = self.bayes4(merge)
        update, _ = self.bayes5(merge)
        state = last_state + update
        state = rearrange(
            state, "(bs k) dim -> bs k dim", bs=batch_size, k=self.num_ensemble
        )
        return state


class ProcessModelsampling(nn.Module):
    """
    process model takes a state or a stack of states (t-n:t-1) and
    predict the next state t. this process model takes in the state and actions
    and outputs a predicted state

    input -> [batch_size, num_ensemble, dim_x]
    action -> [batch_size, num_ensemble, dim_a]
    output ->  [batch_size, num_ensemble, dim_x]
    """

    def __init__(self, num_ensemble, dim_x, dim_a):
        super(ProcessModelsampling, self).__init__()
        self.num_ensemble = num_ensemble
        self.dim_x = dim_x
        self.dim_a = dim_a

        # temporal embedding
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pe = self.positionalencoding1d(128, 4)
        self.pe = self.pe.to(self.device)
        self.freq_list = [50, 30, 10, 5]

        # channel for state variables
        self.bayes1 = LinearFlipout(in_features=self.dim_x, out_features=64)
        self.bayes2 = LinearFlipout(in_features=64, out_features=128)
        self.bayes3 = LinearFlipout(in_features=128, out_features=64)

        # channel for action variables
        self.bayes_a1 = LinearFlipout(in_features=self.dim_a, out_features=64)
        self.bayes_a2 = LinearFlipout(in_features=64, out_features=128)
        self.bayes_a3 = LinearFlipout(in_features=128, out_features=64)

        # channel for adding temporal embedding
        self.bayes_m1 = torch.nn.Linear(128, 128)
        self.bayes_m2 = torch.nn.Linear(128, 64)

        # merge them
        self.bayes4 = LinearFlipout(in_features=64, out_features=32)
        self.bayes5 = LinearFlipout(in_features=32, out_features=self.dim_x)

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

    def forward(self, last_state, action, sample_freq):
        batch_size = last_state.shape[0]
        last_state = rearrange(
            last_state, "bs k dim -> (bs k) dim", bs=batch_size, k=self.num_ensemble
        )
        action = rearrange(
            action, "bs k dim -> (bs k) dim", bs=batch_size, k=self.num_ensemble
        )
        sample_freq = rearrange(
            sample_freq, "bs k dim -> (bs k) dim", bs=batch_size, k=self.num_ensemble
        )

        # branch for the state variables
        x, _ = self.bayes1(last_state)
        x = F.relu(x)
        x, _ = self.bayes2(x)
        x = F.relu(x)
        x, _ = self.bayes3(x)
        x = F.relu(x)

        # branch for the action variables
        y, _ = self.bayes_a1(action)
        y = F.relu(y)
        y, _ = self.bayes_a2(y)
        y = F.relu(y)
        y, _ = self.bayes_a3(y)
        y = F.relu(y)

        # merge branch
        merge = torch.cat((x, y), axis=1)

        # apply temporal embedding
        for k in range(sample_freq.shape[0]):
            freq = int(sample_freq[k])
            ind = self.freq_list.index(freq)
            merge[k] = self.pe[ind, :] + merge[k]

        merge = self.bayes_m1(merge)
        merge = self.bayes_m2(merge)
        feat_map = merge

        merge, _ = self.bayes4(merge)
        update, _ = self.bayes5(merge)
        state = last_state + update
        state = rearrange(
            state, "(bs k) dim -> bs k dim", bs=batch_size, k=self.num_ensemble
        )

        return state, feat_map


class ObservationModel(nn.Module):
    """
    observation model takes a predicted state at t-1 and
    predict the corresponding oberservations. typically, the observation is part of the
    state (H as an identity matrix), unless we are using some observations indirectly to
    update the state

    input -> [batch_size, num_ensemble, dim_x]
    output ->  [batch_size, num_ensemble, dim_z]
    """

    def __init__(self, num_ensemble, dim_x, dim_z):
        super(ObservationModel, self).__init__()
        self.num_ensemble = num_ensemble
        self.dim_x = dim_x
        self.dim_z = dim_z

        self.linear1 = torch.nn.Linear(self.dim_x, 64)
        self.linear2 = torch.nn.Linear(64, 128)
        self.linear3 = torch.nn.Linear(128, 128)
        self.linear4 = torch.nn.Linear(128, 64)
        self.linear5 = torch.nn.Linear(64, self.dim_z)

    def forward(self, state):
        batch_size = state.shape[0]
        state = rearrange(
            state, "bs k dim -> (bs k) dim", bs=batch_size, k=self.num_ensemble
        )
        x = self.linear1(state)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
        x = self.linear3(x)
        x = F.relu(x)
        x = self.linear4(x)
        x = F.relu(x)
        z_pred = self.linear5(x)
        z_pred = rearrange(
            z_pred, "(bs k) dim -> bs k dim", bs=batch_size, k=self.num_ensemble
        )
        return z_pred


class MCLayer(nn.Module):
    """Custom Linear layer but mimics a standard linear layer"""

    def __init__(self, size_in, size_out):
        super().__init__()
        self.size_in, self.size_out = size_in, size_out
        weights = torch.Tensor(size_out, size_in)
        self.weights = nn.Parameter(
            weights
        )  # nn.Parameter is a Tensor that's a module parameter.
        bias = torch.Tensor(size_out)
        self.bias = nn.Parameter(bias)

        # initialize weights and biases
        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))  # weight init
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)  # bias init

    def forward(self, x, mask):
        tmp = self.weights * mask.t()
        w_times_x = torch.mm(x, tmp.t())
        return torch.add(w_times_x, self.bias)  # w times x + b


class BayesianSensorModel(nn.Module):
    """
    the sensor model takes the current raw sensor (usually high-dimensional images)
    and map the raw sensor to low-dimension
    Many advanced model architecture can be explored here, i.e., Vision transformer, FlowNet,
    RAFT, and ResNet families, etc.

    input -> [batch_size, 1, raw_input]
    output ->  [batch_size, num_ensemble, dim_z]
    """

    def __init__(self, num_ensemble, dim_z):
        super(BayesianSensorModel, self).__init__()
        self.dim_z = dim_z
        self.num_ensemble = num_ensemble

        self.mc_layer = MCLayer(30, 128)
        self.fc2 = nn.Linear(128, 512)
        self.fc3 = LinearFlipout(512, 1024)
        self.fc4 = LinearFlipout(1024, 2048)
        self.fc5 = LinearFlipout(2048, 64)
        self.fc6 = LinearFlipout(64, self.dim_z)

    def forward(self, x, mask):
        batch_size = x.shape[0]
        x = rearrange(x, "bs k dim -> (bs k) dim")
        x = repeat(x, "bs dim -> bs k dim", k=self.num_ensemble)
        x = rearrange(x, "bs k dim -> (bs k) dim")

        x = self.mc_layer(x, mask)
        x = F.leaky_relu(x)
        x = self.fc2(x)
        x = F.leaky_relu(x)
        x, _ = self.fc3(x)
        x = F.leaky_relu(x)
        x, _ = self.fc4(x)
        x = F.leaky_relu(x)
        x, _ = self.fc5(x)
        x = F.leaky_relu(x)
        encoding = x
        obs, _ = self.fc6(x)
        obs = rearrange(
            obs, "(bs k) dim -> bs k dim", bs=batch_size, k=self.num_ensemble
        )
        obs_z = torch.mean(obs, axis=1)
        obs_z = rearrange(obs_z, "bs (k dim) -> bs k dim", k=1)
        encoding = rearrange(
            encoding, "(bs k) dim -> bs k dim", bs=batch_size, k=self.num_ensemble
        )
        encoding = torch.mean(encoding, axis=1)
        encoding = rearrange(encoding, "(bs k) dim -> bs k dim", bs=batch_size, k=1)
        return obs, obs_z, encoding


class ObservationNoise(nn.Module):
    def __init__(self, dim_z, r_diag):
        """
        observation noise model is used to learn the observation noise covariance matrix
        R from the learned observation, kalman filter require a explicit matrix for R
        therefore we construct the diag of R to model the noise here

        input -> [batch_size, 1, encoding/dim_z]
        output -> [batch_size, dim_z, dim_z]
        """
        super(ObservationNoise, self).__init__()
        self.dim_z = dim_z
        self.r_diag = r_diag

        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, self.dim_z)

    def forward(self, inputs):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        batch_size = inputs.shape[0]
        constant = np.ones(self.dim_z) * 1e-3
        init = np.sqrt(np.square(self.r_diag) - constant)
        diag = self.fc1(inputs)
        diag = F.relu(diag)
        diag = self.fc2(diag)
        diag = torch.square(diag + torch.Tensor(constant).to(device)) + torch.Tensor(
            init
        ).to(device)
        diag = rearrange(diag, "bs k dim -> (bs k) dim")
        R = torch.diag_embed(diag)
        return R


class Forward_model(nn.Module):
    def __init__(self, num_ensemble, dim_x, dim_a):
        super(Forward_model, self).__init__()
        self.num_ensemble = num_ensemble
        self.dim_x = dim_x
        self.dim_a = dim_a
        self.process_model = ProcessModelsampling(
            self.num_ensemble, self.dim_x, self.dim_a
        )

    def forward(self, states, action, sample_freq):
        state_old, m_state = states
        state_pred = self.process_model(state_old, action, sample_freq)
        m_A = torch.mean(state_pred, axis=1)
        m_state_pred = rearrange(m_A, "bs (k dim) -> bs k dim", k=1)
        output = (state_pred, m_state_pred)
        return output


class Forward_model_stable(nn.Module):
    def __init__(self, num_ensemble, dim_x, dim_a):
        super(Forward_model_stable, self).__init__()
        self.num_ensemble = num_ensemble
        self.dim_x = dim_x
        self.dim_a = dim_a
        self.process_model = ProcessModelAction(
            self.num_ensemble, self.dim_x, self.dim_a
        )

    def forward(self, states, action):
        state_old, m_state = states
        state_pred = self.process_model(state_old, action)
        m_A = torch.mean(state_pred, axis=1)
        m_state_pred = rearrange(m_A, "bs (k dim) -> bs k dim", k=1)
        output = (state_pred, m_state_pred)
        return output


class Ensemble_KF_low(nn.Module):
    def __init__(self, num_ensemble, dim_x, dim_z, dim_a):
        super(Ensemble_KF_low, self).__init__()
        self.num_ensemble = num_ensemble
        self.dim_x = dim_x
        self.dim_z = dim_z
        self.dim_a = dim_a
        self.r_diag = np.ones((self.dim_z)).astype(np.float32) * 0.05
        self.r_diag = self.r_diag.astype(np.float32)

        # instantiate model
        # self.process_model = ProcessModel(self.num_ensemble, self.dim_x)
        self.process_model = ProcessModelAction(
            self.num_ensemble, self.dim_x, self.dim_a
        )
        self.observation_model = ObservationModel(
            self.num_ensemble, self.dim_x, self.dim_z
        )
        self.observation_noise = ObservationNoise(self.dim_z, self.r_diag)
        self.sensor_model = BayesianSensorModel(self.num_ensemble, self.dim_z)

    def forward(self, inputs, states, mask):
        # decompose inputs and states
        batch_size = inputs[0].shape[0]
        action, raw_obs = inputs
        state_old, m_state = states

        ##### prediction step #####
        state_pred = self.process_model(state_old, action)
        m_A = torch.mean(state_pred, axis=1)
        mean_A = repeat(m_A, "bs dim -> bs k dim", k=self.num_ensemble)
        A = state_pred - mean_A
        A = rearrange(A, "bs k dim -> bs dim k")

        ##### update step #####
        H_X = self.observation_model(state_pred)
        mean = torch.mean(H_X, axis=1)
        H_X_mean = rearrange(mean, "bs (k dim) -> bs k dim", k=1)
        m = repeat(mean, "bs dim -> bs k dim", k=self.num_ensemble)
        H_A = H_X - m
        # transpose operation
        H_XT = rearrange(H_X, "bs k dim -> bs dim k")
        H_AT = rearrange(H_A, "bs k dim -> bs dim k")

        # get learned observation
        ensemble_z, z, encoding = self.sensor_model(raw_obs, mask)
        y = rearrange(ensemble_z, "bs k dim -> bs dim k")
        R = self.observation_noise(encoding)

        # measurement update
        innovation = (1 / (self.num_ensemble - 1)) * torch.matmul(H_AT, H_A) + R
        inv_innovation = torch.linalg.inv(innovation)
        K = (1 / (self.num_ensemble - 1)) * torch.matmul(
            torch.matmul(A, H_A), inv_innovation
        )
        gain = rearrange(torch.matmul(K, y - H_XT), "bs dim k -> bs k dim")
        state_new = state_pred + gain

        # gather output
        m_state_new = torch.mean(state_new, axis=1)
        m_state_new = rearrange(m_state_new, "bs (k dim) -> bs k dim", k=1)
        m_state_pred = rearrange(m_A, "bs (k dim) -> bs k dim", k=1)
        encoding = rearrange(encoding, "bs k dim -> (bs k) dim", k=1)
        output = (
            state_new.to(dtype=torch.float32),
            m_state_new.to(dtype=torch.float32),
            m_state_pred.to(dtype=torch.float32),
            z.to(dtype=torch.float32),
            ensemble_z.to(dtype=torch.float32),
            H_X_mean.to(dtype=torch.float32),
            encoding.to(dtype=torch.float32),
        )
        return output


class Ensemble_KF_general(nn.Module):
    def __init__(self, num_ensemble, dim_x, dim_z, dim_a):
        super(Ensemble_KF_general, self).__init__()
        self.num_ensemble = num_ensemble
        self.dim_x = dim_x
        self.dim_z = dim_z
        self.dim_a = dim_a
        self.r_diag = np.ones((self.dim_z)).astype(np.float32) * 0.05
        self.r_diag = self.r_diag.astype(np.float32)

        # instantiate model
        self.process_model = ProcessModelsampling(
            self.num_ensemble, self.dim_x, self.dim_a
        )
        self.observation_model = ObservationModel(
            self.num_ensemble, self.dim_x, self.dim_z
        )
        self.observation_noise = ObservationNoise(self.dim_z, self.r_diag)
        self.sensor_model = BayesianSensorModel(self.num_ensemble, self.dim_z)

    def forward(self, inputs, states, mask):
        # decompose inputs and states
        batch_size = inputs[0].shape[0]
        action, raw_obs, sample_freq = inputs
        state_old, m_state = states

        ##### prediction step #####
        state_pred, feat_map = self.process_model(state_old, action, sample_freq)
        m_A = torch.mean(state_pred, axis=1)
        mean_A = repeat(m_A, "bs dim -> bs k dim", k=self.num_ensemble)
        A = state_pred - mean_A
        A = rearrange(A, "bs k dim -> bs dim k")

        ##### update step #####
        H_X = self.observation_model(state_pred)
        mean = torch.mean(H_X, axis=1)
        H_X_mean = rearrange(mean, "bs (k dim) -> bs k dim", k=1)
        m = repeat(mean, "bs dim -> bs k dim", k=self.num_ensemble)
        H_A = H_X - m
        # transpose operation
        H_XT = rearrange(H_X, "bs k dim -> bs dim k")
        H_AT = rearrange(H_A, "bs k dim -> bs dim k")

        # get learned observation
        ensemble_z, z, encoding = self.sensor_model(raw_obs, mask)
        y = rearrange(ensemble_z, "bs k dim -> bs dim k")
        R = self.observation_noise(encoding)

        # measurement update
        innovation = (1 / (self.num_ensemble - 1)) * torch.matmul(H_AT, H_A) + R
        inv_innovation = torch.linalg.inv(innovation)
        K = (1 / (self.num_ensemble - 1)) * torch.matmul(
            torch.matmul(A, H_A), inv_innovation
        )
        gain = rearrange(torch.matmul(K, y - H_XT), "bs dim k -> bs k dim")
        state_new = state_pred + gain

        # gather output
        m_state_new = torch.mean(state_new, axis=1)
        m_state_new = rearrange(m_state_new, "bs (k dim) -> bs k dim", k=1)
        m_state_pred = rearrange(m_A, "bs (k dim) -> bs k dim", k=1)
        # encoding = rearrange(m_A, "bs k dim -> (bs k) dim", k=1)
        output = (
            state_new.to(dtype=torch.float32),
            m_state_new.to(dtype=torch.float32),
            m_state_pred.to(dtype=torch.float32),
            z.to(dtype=torch.float32),
            ensemble_z.to(dtype=torch.float32),
            H_X_mean.to(dtype=torch.float32),
            feat_map.to(dtype=torch.float32),
            # encoding.to(dtype=torch.float32),
        )
        return output


class Ensemble_KF_no_action(nn.Module):
    def __init__(self, num_ensemble, dim_x, dim_z, dim_a):
        super(Ensemble_KF_no_action, self).__init__()
        self.num_ensemble = num_ensemble
        self.dim_x = dim_x
        self.dim_z = dim_z
        self.dim_a = dim_a
        self.r_diag = np.ones((self.dim_z)).astype(np.float32) * 0.1
        self.r_diag = self.r_diag.astype(np.float32)

        # instantiate model
        self.process_model = ProcessModel(self.num_ensemble, self.dim_x)
        # self.process_model = ProcessModelAction(self.num_ensemble, self.dim_x, self.dim_a)
        self.observation_model = ObservationModel(
            self.num_ensemble, self.dim_x, self.dim_z
        )
        self.observation_noise = ObservationNoise(self.dim_z, self.r_diag)
        self.sensor_model = BayesianSensorModel(self.num_ensemble, self.dim_z)

    def forward(self, inputs, states, mask):
        # decompose inputs and states
        batch_size = inputs[0].shape[0]
        action, raw_obs = inputs
        state_old, m_state = states

        ##### prediction step #####
        state_pred = self.process_model(state_old)
        m_A = torch.mean(state_pred, axis=1)
        mean_A = repeat(m_A, "bs dim -> bs k dim", k=self.num_ensemble)
        A = state_pred - mean_A
        A = rearrange(A, "bs k dim -> bs dim k")

        ##### update step #####
        H_X = self.observation_model(state_pred)
        mean = torch.mean(H_X, axis=1)
        H_X_mean = rearrange(mean, "bs (k dim) -> bs k dim", k=1)
        m = repeat(mean, "bs dim -> bs k dim", k=self.num_ensemble)
        H_A = H_X - m
        # transpose operation
        H_XT = rearrange(H_X, "bs k dim -> bs dim k")
        H_AT = rearrange(H_A, "bs k dim -> bs dim k")

        # get learned observation
        ensemble_z, z, encoding = self.sensor_model(raw_obs, mask)
        y = rearrange(ensemble_z, "bs k dim -> bs dim k")
        R = self.observation_noise(encoding)

        # measurement update
        innovation = (1 / (self.num_ensemble - 1)) * torch.matmul(H_AT, H_A) + R
        inv_innovation = torch.linalg.inv(innovation)
        K = (1 / (self.num_ensemble - 1)) * torch.matmul(
            torch.matmul(A, H_A), inv_innovation
        )
        gain = rearrange(torch.matmul(K, y - H_XT), "bs dim k -> bs k dim")
        state_new = state_pred + gain

        # gather output
        m_state_new = torch.mean(state_new, axis=1)
        m_state_new = rearrange(m_state_new, "bs (k dim) -> bs k dim", k=1)
        m_state_pred = rearrange(m_A, "bs (k dim) -> bs k dim", k=1)
        output = (
            state_new.to(dtype=torch.float32),
            m_state_new.to(dtype=torch.float32),
            m_state_pred.to(dtype=torch.float32),
            z.to(dtype=torch.float32),
            ensemble_z.to(dtype=torch.float32),
            H_X_mean.to(dtype=torch.float32),
        )
        return output


############ only for testing ############
# r_diag = np.ones((2)).astype(np.float32) * 0.1
# r_diag = r_diag.astype(np.float32)

# model = Ensemble_KF(32, 5, 2)
# # print('The model:')
# # print(model)
# # pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
# # print('Total number of parameters: ',pytorch_total_params)

# # some dummy input
# state = torch.FloatTensor(8, 32, 5)
# raw_obs = torch.FloatTensor(8, 2, 224, 224)
# obs = torch.FloatTensor(8, 1, 2)

# print("check -------- ",state.dtype)

# m_state = torch.mean(state, axis = 1)

# input_state = (state, m_state)

# model(raw_obs, input_state)
