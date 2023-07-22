import argparse
import logging
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from bayesian_torch.layers.flipout_layers.linear_flipout import LinearFlipout
import torchvision.models as models
from torch.utils.data import Dataset
from optimizer import build_optimizer
from optimizer import build_lr_scheduler
from einops import rearrange, repeat
import numpy as np
import math
import random
import pickle
import pdb
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = "Times New Roman"
plt.rcParams['savefig.dpi'] = 500

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error


class MCLayer(nn.Module):
    """ Custom Linear layer but mimics a standard linear layer """
    def __init__(self, size_in, size_out):
        super().__init__()
        self.size_in, self.size_out = size_in, size_out
        weights = torch.Tensor(size_out, size_in)
        self.weights = nn.Parameter(weights)  # nn.Parameter is a Tensor that's a module parameter.
        bias = torch.Tensor(size_out)
        self.bias = nn.Parameter(bias)

        # initialize weights and biases
        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5)) # weight init
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)  # bias init

    def forward(self, x, mask):
        # print("the mask ",mask.shape)
        # print("the weight ",self.weights.shape)
        tmp = self.weights * mask.t()
        w_times_x= torch.mm(x, tmp.t())
        return torch.add(w_times_x, self.bias)  # w times x + b


class BasicModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.MCdropout = MCLayer(3, 128)
        self.bayes1 = LinearFlipout(in_features=128,out_features=256)
        self.bayes2 = LinearFlipout(in_features=256,out_features=128)
        self.bayes3 = LinearFlipout(in_features=128,out_features=64)
        self.bayes4 = LinearFlipout(in_features=64,out_features=1)

    def forward(self, x, mask):
        x = self.MCdropout(x, mask)
        x, _ = self.bayes1(x)
        x = F.relu(x)
        x, _ = self.bayes2(x)
        x = F.relu(x)
        x, _ = self.bayes3(x)
        x = F.relu(x)
        x, _ = self.bayes4(x)
        return x


def generate_mask(index):
    tmp = np.ones((3,128))
    if len(index)!=0:
        tmp[index,:] = tmp[index,:] * 0
    mask = torch.tensor(tmp, dtype=torch.float32)
    return mask


def generate_dataset():
    num_point = 1000000
    t = np.random.uniform(0,np.pi/2,(num_point,1))
    # t = np.linspace(0,np.pi/2,num_point).reshape(num_point,1)
    x = np.sin(13*t + 10)
    y = np.sin(7*t - 20)
    z = np.sin(7*t + 5)
    gt = 3*x + 4*y + 5*z
    # print(gt.shape)
    inputs = np.concatenate((x,y,z), axis = 1)
    # print(inputs.shape)

    # standardization 
    f_m = np.mean(inputs,axis=0)
    f_std = np.std(inputs,axis=0)
    g_m = np.mean(gt)
    g_std = np.std(gt)

    inputs = (inputs-f_m)/f_std
    gt = (gt-g_m)/g_std

    # selection_test = random.sample(range(0, num_point), 300)
    # selection_train = random.sample(range(0, num_point), num_point)
    # selection_train, _ = list(set(selection_train) - set(selection_test)),list(set(selection_test) - set(selection_train))


    data = {}
    data['input'] = inputs[300:,:]
    data['gt'] = gt[300:,:]
    data['t'] = t[300:,:]
    with open('toy_train.pkl', 'wb') as f:
        pickle.dump(data, f)

    data = {}
    data['input'] = inputs[:300,:]
    data['gt'] = gt[:300,:]
    data['t'] = t[:300,:]
    with open('toy_test.pkl', 'wb') as f:
        pickle.dump(data, f)
    # if we want to plot
    gt = gt[:300,:]
    # t = np.linspace(1, gt.shape[0], gt.shape[0])
    t = t[:300,:]
    fig = plt.figure()
    plt.scatter(t, gt.flatten(), linewidth=1,label = 'gt')
    plt.legend(loc='upper left')
    plt.show()
    return


class smallDataset(Dataset):
    # Basic Instantiation
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.dataset = pickle.load(open(self.dataset_path, 'rb'))
        self.dataset_length = self.dataset['gt'].shape[0]

    # Length of the Dataset
    def __len__(self):
        # self.dataset_length = 50
        return self.dataset_length-1

    # Fetch an item from the Dataset
    def __getitem__(self, idx):
        inputs = torch.tensor(self.dataset['input'][idx], dtype=torch.float32)
        gt = torch.tensor(self.dataset['gt'][idx], dtype=torch.float32)
        t = torch.tensor(self.dataset['t'][idx], dtype=torch.float32)
        return inputs, gt, t

def train():
    model = BasicModel()
    num_epoch = 20
    mse_criterion = nn.MSELoss()
    dataset = smallDataset('toy_train.pkl')
    dataloader = torch.utils.data.DataLoader(dataset, batch_size = 64,
                                    shuffle=True, num_workers=1)
    optimizer_ = build_optimizer(model, '_', 'adamw', 1e-4, 1e-2, 1e-3)
    num_total_steps = num_epoch * len(dataloader)
    scheduler = build_lr_scheduler(optimizer_, 'polynomial_decay',1e-4, num_total_steps,-1.)

    # Epoch calculations
    global_step = 0
    steps_per_epoch = len(dataloader)
    num_total_steps = num_epoch * steps_per_epoch
    epoch = global_step // steps_per_epoch
    duration = 0
    ####################################################################################################
    # MAIN TRAINING LOOP
    ####################################################################################################
    while epoch < num_epoch:
        step = 0
        for inputs, gt, t in dataloader:
            # define the training curriculum
            optimizer_.zero_grad()
            modalities = [0,1,2]
            index = []
            # index = random.sample(modalities,random.randint(0, 1))
            mask = generate_mask(index)
            out = model(inputs, mask)
            loss = mse_criterion(out, gt)
            # back prop
            loss.backward()
            optimizer_.step()
            current_lr = optimizer_.param_groups[0]['lr']
            # verbose
            if global_step % 1000 == 0:
                string = '[epoch][s/s_per_e/gs]: [{}][{}/{}/{}], lr: {:.12f}, loss: {:.12f}'
                logger.info(string.format(epoch, step, steps_per_epoch, global_step, current_lr, loss))
                if np.isnan(loss.cpu().item()):
                    logger.warning('NaN in loss occurred. Aborting training.')
                    return -1
            step += 1
            global_step += 1
            if scheduler is not None:
                scheduler.step(global_step)
        # Save a model based of a chosen save frequency
        if (epoch == num_epoch-1):
            checkpoint = {'global_step': global_step,
                            'model': model.state_dict(),
                            'optimizer': optimizer_.state_dict()}
            torch.save(checkpoint,
                        os.path.join('toy_model-{}'.format(global_step)))
        # Update epoch
        epoch += 1

def test(checkpoint_path):
    model = BasicModel()
    # Load the pretrained model
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    dataset = smallDataset('toy_test.pkl')
    test_dataloader = torch.utils.data.DataLoader(dataset, batch_size = 1,
                                    shuffle=False, num_workers=1)

    index = [1,2] # 1 or 2 to select y or z to remove from the input
    mask = generate_mask(index)
    data = {}
    data_save = []
    gt_save = []
    t_save = []
    for inputs, gt, t in test_dataloader:
        with torch.no_grad():
            out = model(inputs, mask)

            pred_final = out
            gt_final = gt
            pred_final = pred_final.cpu().detach().numpy()
            gt_final = gt_final.cpu().detach().numpy()
            t_final = t.cpu().detach().numpy()
            data_save.append(pred_final)
            gt_save.append(gt_final)
            t_save.append(t_final)


    data['pred'] = data_save
    data['gt'] = gt_save
    data['t'] = t_save
    save_path = 'test_rm_yz.pkl'
    with open(save_path, 'wb') as f:
        pickle.dump(data, f)


def visualization():
    # visual_list = ['test_rm_none.pkl', 'test_rm_x.pkl', 'test_rm_y.pkl', 'test_rm_z.pkl', 'test_rm_a.pkl']
    # visual_list = ['test_rm_xy.pkl', 'test_rm_xz.pkl', 'test_rm_xa.pkl', 'test_rm_yz.pkl'
    #                 ,'test_rm_ya.pkl', 'test_rm_za.pkl']
    visual_list = ['test_rm_none.pkl','test_rm_xy.pkl', 'test_rm_xz.pkl', 'test_rm_yz.pkl']
    size_1 = len(visual_list)
    size_2 = 1
    fig = plt.figure(figsize=(size_1, size_2))
    ids = 1
    for name in visual_list:
        with open(name, 'rb') as f:
            data = pickle.load(f)
            test_demo = data['pred']
            gt_data = data['gt']
            t = data['t']
        test_demo = np.array(test_demo).reshape(299,1)
        gt_data = np.array(gt_data).reshape(299,1)
        t = np.array(t).reshape(299,1)

        rmse = mean_squared_error(test_demo, gt_data, squared=False)
        mae = mean_absolute_error(test_demo, gt_data)
        print(rmse, ' ', mae)

        # if we want to plot
        plt.subplot(size_1, size_2, ids)
        num_point = 100
        # t = np.linspace(1, num_point, num_point)
        plt.scatter(t[:num_point], gt_data[:num_point].flatten(), color = 'r',linewidth=1.5,label = 'gt', alpha = 0.8)
        plt.scatter(t[:num_point], test_demo[:num_point].flatten(), color = 'b',linewidth=1.5,label = 'pred', alpha = 0.6)
        plt.legend(loc='upper right')
        plt.title(name)
        ids = ids + 1
    plt.show()

def func(input):
    return input[0]**2+input[1]


def real_time_test(checkpoint_path):
    model = BasicModel()
    # Load the pretrained model
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model'])
    model.eval()

    index = [] # 1 or 2 to select y or z to remove from the input
    mask = generate_mask(index)

    num_point = 10
    x = np.random.uniform(-10,10,(num_point,1))
    z = np.random.uniform(-20,20,(num_point,1))
    y = 2*z**2
    a = np.random.uniform(-1,1,(num_point,1))
    gt = np.square(x) + y
    # print(gt.shape)
    inputs = np.concatenate((x,y,z,a), axis = 1)
    # print(inputs.shape)

    # standardization 
    f_m = np.mean(inputs,axis=0)
    f_std = np.std(inputs,axis=0)
    g_m = np.mean(gt)
    g_std = np.std(gt)

    inputs = (inputs-f_m)/f_std
    gt = (gt-g_m)/g_std

    random_input = np.array([1,8,2,1])
    target = func(random_input)
    print(target)
    random_input = (random_input-f_m)/f_std
    target = (target-g_m)/g_std

    # for i in range (gt.shape[0]):
    print('----')
    x = torch.tensor(random_input.reshape(1,4), dtype=torch.float32)
    t = torch.tensor(target.reshape(1,1), dtype=torch.float32)
    out = model(x, mask)
    print(x)
    print(t)
    print(out)
    print('----')
    print(t*g_std+g_m)
    print(out*g_std+g_m)


logging_kwargs = dict(
        level="INFO",
        format="%(asctime)s %(threadName)s %(levelname)s %(name)s - %(message)s",
        style='%',
    )
logging.basicConfig(**logging_kwargs)
logger = logging.getLogger('toy')


def main():
    # generate_dataset()
    # train()
    # test('toy_model-312420')
    visualization()
    # real_time_test('toy_model-15200')





if __name__ == "__main__":
    main()