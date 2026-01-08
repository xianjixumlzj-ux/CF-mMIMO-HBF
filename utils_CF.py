import torch.utils.data as data
from termcolor import colored
import torch.nn.functional as F
import torch
import scipy.io
from numpy import genfromtxt
import numpy as np
from utils_math import Th_comp_matmul, Th_inv, Th_pinv
import neptune
import re
import torch.nn as nn
import time
import math

# Database ####################################################################################################################
class Data_Reader(data.Dataset):
    def __init__(self, filename, Us, Mr, Nrf, K, N_BS, include_delay=False):
        print(colored('You select Extended dataset', 'cyan'))
        print(colored(filename, 'yellow'), 'is loading ... ')
        np_data = np.load(filename, allow_pickle=True)
        self.assoc = None
        self.assoc_feature_dim = 0
        channel_len = Us * Mr * N_BS
        rssi_len = Us * N_BS * K

        if isinstance(np_data, np.lib.npyio.NpzFile):
            self.channel = np_data['channel']
            self.RSSI_N = np_data['rssi'].real.astype(float)
            if 'assoc' in np_data:
                self.assoc = np_data['assoc'].astype(float)
        else:
            self.channel = np_data[:, 0:channel_len]
            self.RSSI_N = np_data[:, channel_len:channel_len + rssi_len].real.astype(float)
            if np_data.shape[1] > channel_len + rssi_len:
                self.assoc = np_data[:, channel_len + rssi_len:].astype(float)

        if self.assoc is not None:
            assoc_flat = self.assoc.reshape(self.assoc.shape[0], -1)
            self.assoc_feature_dim = assoc_flat.shape[1] // (Us * N_BS)

        self.n_samples = self.channel.shape[0]
        np_data = np.load(filename)
        channel_len = Us * Mr * N_BS
        rssi_len = Us * N_BS * K
        delay_len = Us * N_BS
        self.channel = np_data[:, 0:channel_len]
        self.RSSI_N = np_data[:, channel_len:channel_len + rssi_len].real.astype(float)
        self.delay = None
        if np_data.shape[1] >= channel_len + rssi_len + delay_len:
            self.delay = np_data[:, channel_len + rssi_len:channel_len + rssi_len + delay_len].astype(float)
        self.include_delay = include_delay
        self.n_samples = np_data.shape[0]
        if isinstance(np_data, np.lib.npyio.NpzFile):
            if "channel_flat" not in np_data or "rssi_flat" not in np_data:
                raise ValueError("NPZ格式缺少channel_flat或rssi_flat字段，无法读取")
            self.channel = np_data["channel_flat"]
            self.RSSI_N = np_data["rssi_flat"].real.astype(float)
            self.n_samples = self.channel.shape[0]
        else:
            self.channel = np_data[:, 0:Us * Mr * N_BS]
            self.RSSI_N = np_data[:, Us * Mr * N_BS:].real.astype(float)
            self.n_samples = np_data.shape[0]

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        channel = torch.tensor(self.channel[index]).type(torch.complex64)
        rssi = torch.tensor(self.RSSI_N[index])
        if self.include_delay and self.delay is not None:
            delay = torch.tensor(self.delay[index]).type(torch.float32)
            return channel, rssi, delay
        return channel, rssi
        if self.assoc is None:
            return channel, rssi
        assoc = torch.tensor(self.assoc[index]).float()
        return channel, rssi, assoc


# readme reader for HBF initial parameters ####################################################################################
def md_reader(DB_name):
    md = genfromtxt(''.join((DB_name, '/DATASET.md')), delimiter='\n', dtype='str')
    Us = int(re.findall(r'\d+', md[1])[0])
    Mr = int(re.findall(r'\d+', md[2])[0])
    Nrf = int(re.findall(r'\d+', md[3])[0])
    N_BS = int(re.findall(r'\d+', md[4])[0])
    K = int(re.findall(r'\d+', md[5])[0])
    Noise_pwr = 10 ** -(int(re.findall(r'\d+', md[7])[0]) / 10)
    return Us, Mr, Nrf, N_BS, K, Noise_pwr

class Initialization_Model_Params(object):
    def __init__(self,
                 DB_name,
                 Us,
                 Mr,
                 Nrf,
                 K,
                 K_limited,
                 N_BS,
                 Noise_pwr,
                 device,
                 device_ids
                 ):
        self.DB_name = DB_name
        self.Us = Us
        self.Mr = Mr
        self.Nrf = Nrf
        self.K = K
        self.K_limited = K_limited
        self.N_BS = N_BS
        self.Noise_pwr = Noise_pwr
        self.device = device
        self.dev_id = device_ids

    def Data_Load(self, include_delay=False):
        DataBase = Data_Reader(''.join((self.DB_name, '/dataSet_130.npy')), self.Us, self.Mr, self.Nrf, self.K, self.N_BS, include_delay=include_delay)
    def Data_Load(self):
        DataBase = Data_Reader(''.join((self.DB_name, '/dataSet_130.npy')), self.Us, self.Mr, self.Nrf, self.K, self.N_BS)
        return DataBase  # , uniq_dis_label

    def Code_Read(self):
        mat_C1 = torch.tensor(np.array(scipy.io.loadmat('dataSet4x64x8x4/BS4/Codebook_SSB.mat')['codebook'])).type(torch.complex64)
        mat_C2 = torch.tensor(np.array(scipy.io.loadmat('dataSet4x64x8x4/BS5/Codebook_SSB.mat')['codebook'])).type(torch.complex64)
        mat_C3 = torch.tensor(np.array(scipy.io.loadmat('dataSet4x64x8x4/BS8/Codebook_SSB.mat')['codebook'])).type(torch.complex64)
        mat_C4 = torch.tensor(np.array(scipy.io.loadmat('dataSet4x64x8x4/BS9/Codebook_SSB.mat')['codebook'])).type(torch.complex64)
        return [mat_C1[0:2, :], mat_C2[0:6, :], mat_C3[0:6, :], mat_C4[0:2, :]], \
            [len(mat_C1[0:2, :]), len(mat_C2[0:6, :]), len(mat_C3[0:6, :]), len(mat_C4[0:2, :])]

    def plot_grad_flow(self, named_parameters):
        # ave_grads = []
        for n, p in named_parameters:
            if(p.requires_grad) and ("bias" not in n):
                # ave_grads.append(p.grad.abs().mean())
                neptune.send_metric(f'layer{n}', p.grad.abs().mean())


class Loss_FCDP_Rate_Based(torch.nn.Module):
    def __init__(self, Us, Mr, Nrf, N_BS, Noise_pwr):
        super(Loss_FCDP_Rate_Based, self).__init__()
        self.Us = Us
        self.Mr = Mr
        self.Nrf = Nrf
        self.N_BS = N_BS
        self.noise_power = Noise_pwr

    def rate_calculator(self, FDP, channel):
        W = torch.abs(torch.matmul(torch.conj(channel), FDP).sum(1)) ** 2
        SINR = torch.diagonal(W, dim1=1, dim2=2) / (torch.sum(W, 2) - torch.diagonal(W, dim1=1, dim2=2) + self.noise_power)
        userRates = torch.log2(1 + SINR)
        sumRate = userRates.sum(1)
        avgRate = userRates.mean(1)
        return sumRate, avgRate

    def forward(self, FDP, channel):
        FDP = FDP / torch.linalg.norm(FDP, dim=1, keepdim=True)
        FDP = FDP.view(-1, self.Us, self.Mr, self.N_BS).permute(0, 3, 2, 1)
        sum_rate, _ = Loss_FCDP_Rate_Based.rate_calculator(self, FDP, channel)
        return - sum_rate.mean()

    def evaluate_rate(self, FDP, channel):
        FDP = FDP / torch.linalg.norm(FDP, dim=1, keepdim=True)
        FDP = FDP.view(-1, self.Us, self.Mr, self.N_BS).permute(0, 3, 2, 1)
        sum_rate, avg_rate = Loss_FCDP_Rate_Based.rate_calculator(self, FDP, channel)
        return sum_rate.mean(), avg_rate.mean()

class Loss_HCBF_Rate_Based(torch.nn.Module):
    def __init__(self, Us, Mr, Nrf, N_BS, Noise_pwr, enable_async=False, fc=28e9):
        super(Loss_HCBF_Rate_Based, self).__init__()
        self.Us = Us
        self.Mr = Mr
        self.Nrf = Nrf
        self.N_BS = N_BS
        self.noise_power = Noise_pwr
        self.enable_async = enable_async
        self.fc = fc

    def _apply_async_phase(self, channel, delay, enable_async=None, fc=None):
        if delay is None:
            return channel
        if enable_async is None:
            enable_async = self.enable_async
        if not enable_async:
            return channel
        if fc is None:
            fc = self.fc
        delay = delay.to(channel.device)
        phi = torch.exp(-1j * 2 * math.pi * fc * delay)
        return channel * phi.unsqueeze(-1)

    def rate_calculator(self, FDP, channel, delay=None, enable_async=None, fc=None):
        channel = self._apply_async_phase(channel, delay, enable_async, fc)
        W = torch.abs(torch.matmul(torch.conj(channel), FDP).sum(1)) ** 2
        SINR = torch.diagonal(W, dim1=1, dim2=2) / (torch.sum(W, 2) - torch.diagonal(W, dim1=1, dim2=2) + self.noise_power)
        userRates = torch.log2(1 + SINR)
        sumRate = userRates.sum(1)
        avgRate = userRates.mean(1)
        return sumRate, avgRate, userRates

    def rate_calculator_4d(self, FDP, channel, delay=None, enable_async=None, fc=None):
        channel = self._apply_async_phase(channel, delay, enable_async, fc)
        W = torch.abs(torch.matmul(torch.conj(channel), FDP).sum(5)) ** 2
        SINR = torch.diagonal(W, dim1=5, dim2=6) / (torch.sum(W, 6) - torch.diagonal(W, dim1=5, dim2=6) + self.noise_power)
        userRates = torch.log2(1 + SINR)
        sumRate = userRates.sum(5)
        avgRate = userRates.mean(5)
        return sumRate, avgRate

    def forward(self, W, channel, A, delay=None, enable_async=None, fc=None):
        HBF = torch.matmul(A.view(-1, len(channel), self.Nrf, self.Mr, self.N_BS).permute(0, 1, 4, 3, 2), W)
        HBF = HBF / torch.unsqueeze(torch.unsqueeze(torch.linalg.norm(HBF.flatten(2), dim=2).unsqueeze(2), 3), 4)
        sum_rate, _ = Loss_HCBF_Rate_Based.rate_calculator_4d(self, HBF, channel, delay=delay, enable_async=enable_async, fc=fc)
        return sum_rate.T

    def evaluate_rate(self, W, channel, A, delay=None, enable_async=None, fc=None):
        HBF = torch.matmul(A.view(len(channel), self.Nrf, self.Mr, self.N_BS).permute(0, 3, 2, 1), W)
        HBF = HBF / torch.unsqueeze(torch.unsqueeze(torch.linalg.norm(HBF.flatten(1), dim=1).unsqueeze(1), 2), 3)
        sum_rate, avgRate = Loss_HCBF_Rate_Based.rate_calculator(self, HBF, channel, delay=delay, enable_async=enable_async, fc=fc)
        return sum_rate.mean(), avgRate.mean()

class Loss_HCBF_S_Rate_Based(torch.nn.Module):
    def __init__(self, Us, Mr, Nrf, N_BS, Noise_pwr, enable_async=False, fc=28e9):
        super(Loss_HCBF_S_Rate_Based, self).__init__()
        self.Us = Us
        self.Mr = Mr
        self.Nrf = Nrf
        self.N_BS = N_BS
        self.noise_power = Noise_pwr
        self.enable_async = enable_async
        self.fc = fc

    def _apply_async_phase(self, channel, delay, enable_async=None, fc=None):
        if delay is None:
            return channel
        if enable_async is None:
            enable_async = self.enable_async
        if not enable_async:
            return channel
        if fc is None:
            fc = self.fc
        delay = delay.to(channel.device)
        phi = torch.exp(-1j * 2 * math.pi * fc * delay)
        return channel * phi.unsqueeze(-1)

    def forward(self, W, channel, A_s1, A_s2, A_s3, A_s4, sinr_3d, delay=None, enable_async=None, fc=None):
        A = torch.cat((A_s1.unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, sinr_3d.shape[1], sinr_3d.shape[2], sinr_3d.shape[3], 1, 1, 1),
                       A_s2.unsqueeze(0).unsqueeze(2).unsqueeze(2).repeat(sinr_3d.shape[0], 1, sinr_3d.shape[2], sinr_3d.shape[3], 1, 1, 1),
                       A_s3.unsqueeze(0).unsqueeze(0).unsqueeze(3).repeat(sinr_3d.shape[0], sinr_3d.shape[1], 1, sinr_3d.shape[3], 1, 1, 1),
                       A_s4.unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(sinr_3d.shape[0], sinr_3d.shape[1], sinr_3d.shape[2], 1, 1, 1, 1)), axis=5)

        HBF = torch.matmul(A.view(sinr_3d.shape[0],
                                  sinr_3d.shape[1],
                                  sinr_3d.shape[2],
                                  sinr_3d.shape[3], len(channel), self.N_BS, self.Nrf, self.Mr).permute(0, 1, 2, 3, 4, 5, 7, 6), W)

        power = torch.unsqueeze(torch.unsqueeze(torch.linalg.norm(HBF.flatten(5), dim=5).unsqueeze(5), 6), 7)
        HBF = HBF / power
        channel = self._apply_async_phase(channel, delay, enable_async, fc)
        sinr_3d = Loss_HCBF_Rate_Based.rate_calculator_4d(self, HBF, channel)[0]
        return sinr_3d, power

    def evaluate_rate(self, W, channel, A, delay=None, enable_async=None, fc=None):
        HBF = torch.matmul(A.view(len(channel), self.N_BS, self.Nrf, self.Mr).permute(0, 1, 3, 2), W)
        Power = torch.unsqueeze(torch.unsqueeze(torch.linalg.norm(HBF.flatten(1), dim=1).unsqueeze(1), 2), 3)
        HBF = HBF / Power
        sum_rate, avgRate, userRates = Loss_HCBF_Rate_Based.rate_calculator(self, HBF, channel, delay=delay, enable_async=enable_async, fc=fc)
        return sum_rate.mean(), avgRate.mean(), userRates, Power.mean()


def FLP_loss(x, y):
    log_prob = - 1.0 * F.softmax(x, 1)
    temp = log_prob * y
    cel = temp.sum(dim=1)
    cel = cel.mean()
    return cel

def FLP_loss_s(x, y):
    log_prob = - 1.0 * x
    temp = log_prob * y
    cel = temp.sum(dim=1).sum(dim=1).sum(dim=1).sum(dim=1)
    cel = cel.mean()
    return cel
