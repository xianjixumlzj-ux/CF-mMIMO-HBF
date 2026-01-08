from networks_arch_CF import Net_FD, Net_PD, Net_FD_CNN, Net_PD_CNN
import torch.nn as nn
import torch


class Networks_activations(object):
    def __init__(self,
                 DB_name,
                 Us,
                 Mr,
                 Nrf,
                 K,
                 K_limited,
                 assoc_feature_dim,
                 N_BS,
                 Noise_pwr,
                 Net_MT_Type,
                 device,
                 device_ids,
                 n_input,
                 n_hidden,
                 n_output_reg,
                 Codebook_len,
                 p_dropout,
                 out_channel,
                 kernel_s,
                 padding
                 ):
        self.DB_name = DB_name
        self.Us = Us
        self.Mr = Mr
        self.Nrf = Nrf
        self.K = K
        self.K_limited = K_limited
        self.assoc_feature_dim = assoc_feature_dim
        self.K_limited_input = K_limited + assoc_feature_dim
        self.N_BS = N_BS
        self.Noise_pwr = Noise_pwr
        self.Net_MT_Type = Net_MT_Type
        self.device = device
        self.dev_id = device_ids
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output_reg = n_output_reg
        self.Codebook_len = Codebook_len
        self.p_dropout = p_dropout
        self.out_channel = out_channel
        self.kernel_s = kernel_s
        self.padding = padding

    def Network_CF(self):
        if self.Net_MT_Type == 2:
            return nn.DataParallel(Net_FD_CNN(1, self.n_hidden, self.n_output_reg, self.Codebook_len,
                                              self.p_dropout, self.Us, self.K_limited_input, self.out_channel, self.kernel_s, self.padding),
                                   device_ids=self.dev_id).to(self.device)
        elif self.Net_MT_Type == 3:
            return nn.DataParallel(Net_PD_CNN(1, self.n_hidden, self.n_output_reg, self.Codebook_len,
                                              self.p_dropout, self.Us, self.K_limited_input, self.out_channel, self.kernel_s, self.padding),
                                   device_ids=self.dev_id).to(self.device)

    def Inp_MT(self, RSSI):
        feature_len = RSSI.shape[1] // (self.Us * self.N_BS)
        RSSI_temp = RSSI.view(-1, self.Us, self.N_BS, feature_len)
        rssi_core = RSSI_temp[:, :, :, 0:self.K_limited]
        if self.assoc_feature_dim > 0 and feature_len >= self.K + self.assoc_feature_dim:
            assoc_start = self.K
            assoc_end = assoc_start + self.assoc_feature_dim
            assoc_features = RSSI_temp[:, :, :, assoc_start:assoc_end]
            Inputs_MT = torch.cat((rssi_core, assoc_features), dim=3)
        else:
            Inputs_MT = rssi_core
        Inputs_MT = Inputs_MT.permute(0, 2, 1, 3).float().to(self.device)
        return Inputs_MT
