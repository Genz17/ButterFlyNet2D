'''
This is the core network implementaion,
Put Non-zero entries in Butterfly Algorithm into CNN.
'''

import sys
import os
sys.path.append(os.path.abspath(os.path.join(__file__,'..','..','Funcs')))
sys.path.append(os.path.abspath(os.path.join(__file__,'..','..','Nets')))
from Chebyshev_Nodes import *
from Lagrange_Interpolation import *
import torch
import numpy as np
import torch.nn as nn

class ButterFlyNet2D(nn.Module):
    '''
    Pay attention to that the paras are x first then y, different to the order in pytorch.
    '''
    def __init__(self, in_channel, height, width,
                 layer_number, chebyshev_number,
                 left_frequency_kx, right_frequency_kx,
                 left_frequency_ky, right_frequency_ky,
                 initMethod, positivereal):
        super(ButterFlyNet2D, self).__init__()

        self.in_channel_num         = in_channel
        self.height                 = height
        self.width                  = width
        self.layer_number           = layer_number
        self.w_height               = int(self.height / (2 ** (self.layer_number - 1)))
        self.w_width                = int(self.width / (2 ** (self.layer_number - 1)))
        self.chebyshev_number       = chebyshev_number
        self.left_frequency_kx      = left_frequency_kx
        self.right_frequency_kx     = right_frequency_kx
        self.frequency_kx_length    = right_frequency_kx - left_frequency_kx
        self.frequency_ky_length    = right_frequency_ky - left_frequency_ky
        self.left_frequency_ky      = left_frequency_ky
        self.right_frequency_ky     = right_frequency_ky
        self.leftover_kx            = self.frequency_kx_length//(2**layer_number)
        self.leftover_ky            = self.frequency_ky_length//(2**layer_number)
        self.frequency_kx_uni_dots  = np.array(range(left_frequency_kx, right_frequency_kx))
        self.frequency_ky_uni_dots  = np.array(range(left_frequency_ky, right_frequency_ky))
        self.plot_x, self.plot_y    = np.meshgrid(self.frequency_kx_uni_dots, self.frequency_ky_uni_dots)
        self.initMethod             = initMethod
        self.positivereal           = positivereal
        self.conv_dict              = self.Generate_Convs()


    def Generate_Convs(self):
        conv_dict_first                     = {str((0, in_channel, 0, 0)):
                                                    nn.Conv2d(in_channels=4,
                                                    out_channels=4 * 4 * (self.chebyshev_number ** 2),
                                                    kernel_size=(self.w_height, self.w_width),
                                                    stride=(self.w_height, self.w_width),
                                                    bias=True)
                                            for in_channel in range(self.in_channel_num)}
        conv_dict_rcs                       ={str((lyr, in_channel, A_y, A_x)):
                                                  nn.Conv2d(in_channels=4 * self.chebyshev_number ** 2,
                                                            out_channels=4 * 4 * (self.chebyshev_number ** 2),
                                                            kernel_size=(2, 2),
                                                            stride=(2, 2),
                                                            bias=True)
                                            for lyr in range(1,self.layer_number)
                                            for in_channel in range(self.in_channel_num)
                                            for A_y in range(2**lyr)
                                            for A_x in range(2**lyr)}
        conv_dict_ft                        = {str((self.layer_number, in_channel, A_y, A_x)):
                                                   nn.Conv2d(in_channels=4 * self.chebyshev_number ** 2,
                                                             out_channels=4 * self.leftover_kx * self.leftover_ky,
                                                             kernel_size=(1, 1),
                                                             bias=True)
                                            for in_channel in range(self.in_channel_num)
                                            for A_y in range(2**self.layer_number)
                                            for A_x in range(2**self.layer_number)}
        conv_dict                   = {}
        conv_dict.update(conv_dict_first)
        conv_dict.update(conv_dict_rcs)
        conv_dict.update(conv_dict_ft)
        if self.initMethod == 'Fourier':
            conv_Weights_dict_first = {str((0, in_channel, 0, 0)):
                                           self.generate_First_Layer_Weights()
                                       for in_channel in range(self.in_channel_num)}
            conv_Weights_dict_rcs = {str((lyr, in_channel, A_y, A_x)):
                                         self.generate_Recursion_Layer_Weights(A_x, A_y, lyr)
                                     for lyr in range(1, self.layer_number)
                                     for in_channel in range(self.in_channel_num)
                                     for A_y in range(2 ** lyr)
                                     for A_x in range(2 ** lyr)}
            conv_Weights_dict_ft = {str((self.layer_number, in_channel, A_y, A_x)):
                                        self.generate_FT_Layer_Weights(A_x, A_y)
                                    for in_channel in range(self.in_channel_num)
                                    for A_y in range(2 ** self.layer_number)
                                    for A_x in range(2 ** self.layer_number)}
            conv_Weights_dict = {}
            conv_Weights_dict.update(conv_Weights_dict_first)
            conv_Weights_dict.update(conv_Weights_dict_rcs)
            conv_Weights_dict.update(conv_Weights_dict_ft)

            for key in conv_dict.keys():
                conv_dict[key].weight   = conv_Weights_dict[key]
                conv_dict[key].bias     = nn.Parameter(torch.zeros_like(conv_dict[key].bias, dtype=torch.float32))
        elif self.initMethod == 'kaimingU':
            for key in conv_dict.keys():
                nn.init.kaiming_uniform_(conv_dict[key].weight)
                nn.init.constant_(conv_dict[key].bias,0.0)
        elif self.initMethod == 'kaimingN':
            for key in conv_dict.keys():
                nn.init.kaiming_normal_(conv_dict[key].weight)
                nn.init.constant_(conv_dict[key].bias,0.0)

        elif self.initMethod == 'orthogonal':
            for key in conv_dict.keys():
                nn.init.orthogonal_(conv_dict[key].weight)
                nn.init.constant_(conv_dict[key].bias,0.0)
        return nn.ModuleDict(conv_dict)

    def forward(self, input_data):
        input_data = self.split(input_data)

        out1 = torch.zeros(input_data.shape[0],
                           self.in_channel_num * 4 * 4 * (self.chebyshev_number ** 2),
                           2**(self.layer_number-1),
                           2**(self.layer_number-1), dtype=torch.float32)
        for channel in range(self.in_channel_num):
            out1[:, channel*4*4*(self.chebyshev_number**2):(channel+1)*4*4*(self.chebyshev_number**2)] = \
                self.conv_dict[str((0, channel, 0, 0))](input_data[:, 4*channel:4*(channel+1)])
        out_rcs = torch.nn.functional.relu(out1)

        for lyr in range(1, self.layer_number):
            out = torch.zeros(input_data.shape[0],
                              self.in_channel_num*4*4**(lyr+1)*(self.chebyshev_number**2),
                              2**(self.layer_number-lyr-1),
                              2**(self.layer_number-lyr-1), dtype=torch.float32)
            for channel in range(self.in_channel_num):
                for A in range(4 ** lyr):
                    A_x = A % (2 ** lyr)
                    A_y = A // (2 ** lyr)
                    out_mid = self.conv_dict[str((lyr, channel, A_y, A_x))](out_rcs[:, (channel*(2**lyr)*(2**lyr)+((A_y*2**lyr)+A_x))*
                                                               4*(self.chebyshev_number**2):
                                                        (channel*(2**lyr)*(2**lyr)+((A_y*2**lyr)+A_x)+1)*
                                                        4*(self.chebyshev_number**2)])

                    for location in range(4):
                        x_location = location % 2
                        y_location = location // 2
                        out[:, (channel*4*(2**lyr)*(2**lyr)+
                                (4*(2**lyr)*A_y+2*A_x+y_location*(2**(lyr+1))+x_location))*
                               4*(self.chebyshev_number**2):
                               (channel*4*(2**lyr)*(2**lyr)+
                                (4*(2**lyr)*A_y+2*A_x+y_location*(2**(lyr+1))+x_location)+1)*
                               4*(self.chebyshev_number**2)] = \
                            out_mid[:, 4*(y_location*2+x_location)*(self.chebyshev_number**2):
                                                    4*(y_location*2+x_location+1)*(self.chebyshev_number**2)]
                    del out_mid

            out_rcs = torch.nn.functional.relu(out)

        out_final = torch.zeros(input_data.shape[0],
                          self.in_channel_num*4*self.frequency_kx_length*self.frequency_ky_length,
                          1,
                          1, dtype=torch.float32)

        for channel in range(self.in_channel_num):
            for A in range(4**(self.layer_number)):
                A_y = A//(2**self.layer_number)
                A_x = A%(2**self.layer_number)
                out_final_mid = self.conv_dict[str((self.layer_number, channel, A_y, A_x))](
                    out_rcs[:, (channel * (4 ** (self.layer_number)) + A_y * (2 ** (self.layer_number)) + A_x) *
                               4 * (self.chebyshev_number ** 2):
                               (channel * (4 ** (self.layer_number)) + A_y * (2 ** (self.layer_number)) + A_x + 1) *
                               4 * (self.chebyshev_number ** 2)])

                for location in range(self.leftover_kx*self.leftover_ky):
                    ky = location//self.leftover_kx
                    kx = location%(self.leftover_kx)
                    out_final[:, 4 *
                                 (channel * self.frequency_ky_length * self.frequency_kx_length +
                                  A_y * (2 ** (self.layer_number)) * self.leftover_ky * self.leftover_kx +
                                  A_x * self.leftover_kx +
                                  ky * self.frequency_kx_length +
                                  kx):
                                 4 *
                                 (channel * self.frequency_ky_length * self.frequency_kx_length +
                                  A_y * (2 ** (self.layer_number)) * self.leftover_ky * self.leftover_kx +
                                  A_x * self.leftover_kx +
                                  ky * self.frequency_kx_length +
                                  kx + 1)] = \
                        out_final_mid[:, 4 * (ky * self.leftover_kx + kx):4 * (ky * self.leftover_kx + kx + 1)]

                del out_final_mid

        out_final = torch.nn.functional.relu(out_final)
        out_joint = self.joint(out_final)
        out_A = out_joint.view(
            (input_data.shape[0], self.in_channel_num, self.frequency_ky_length, self.frequency_kx_length))
        return out_A




    def split(self, input_data):
        data_split = torch.zeros((input_data.shape[0],
                     4*input_data.shape[1],
                     input_data.shape[2],
                     input_data.shape[3]), dtype=torch.float32)

        if self.positivereal:
            for channel in range(input_data.shape[1]):
                data_split[:, 4 * channel, :, :] = \
                    input_data[:, channel, :, :]
        else:
            for channel in range(input_data.shape[1]):
                data_split[:, 4 * channel, :, :] = \
                    input_data[:, channel, :, :].real
                data_split[:, 4 * channel + 1, :, :] = \
                    input_data[:, channel, :, :].imag
                data_split[:, 4 * channel + 2, :, :] = \
                    -input_data[:, channel, :, :].real
                data_split[:, 4 * channel + 3, :, :] = \
                    -input_data[:, channel, :, :].imag
            data_split = torch.nn.functional.relu(data_split)
        return data_split



    def generate_First_Layer_Weights(self):
        # layer 0
        '''
        :param input_data: [w_x * 2^(L-1), w_y * 2^(L-1)],  [batch_size, channel, height, width]
                                                            [batch_size, channel, w_height, w_width]
        see input_data as [batch_size, 4*channel, height, width]
        '''

        single_input_x          = np.linspace(0, (1-1/self.width), self.width)
        single_input_y          = np.linspace(0, (1-1/self.height), self.height)
        xi_x                    = np.array([self.left_frequency_kx + (self.frequency_kx_length/4),
                                            self.left_frequency_kx +(self.frequency_kx_length*3/4)])
        xi_y                    = np.array([self.left_frequency_ky + (self.frequency_ky_length/4),
                                            self.left_frequency_ky + (self.frequency_ky_length*3/4)])
        chebyshev_dots          = Chebyshev_Nodes([0, 1/(2**(self.layer_number-1))], self.chebyshev_number)
        wt_fst_lyr_single       = np.zeros((4*4*(self.chebyshev_number**2),     # split into Re+, Im+, Re-, Im- part.
                                            4,                                  # split into Re+, Im+, Re-, Im- part.
                                            self.w_height,
                                            self.w_width))                      # weight for a single layer input

        beta                    = np.zeros((4*(self.chebyshev_number**2),       # (dots_in_A_x, dots_in_A_y, chebs_B_x, chebs_B_y)
                                            1,                                  # (2,2,r,r)
                                            self.w_height,
                                            self.w_width), dtype=complex)


        '''Generate beta'''
        for dot in range(4*(self.chebyshev_number**2)*self.w_width*self.w_height):
            dot_in_A = dot//((self.chebyshev_number**2)*self.w_width*self.w_height)
            dot_in_A_y = dot_in_A//2
            dot_in_A_x = dot_in_A%2
            leftdots = dot%((self.chebyshev_number**2)*self.w_width*self.w_height)
            chebs = leftdots//(self.w_width*self.w_height)
            chebs_height = chebs//(self.chebyshev_number)
            chebs_width = chebs%(self.chebyshev_number)
            location = leftdots % (self.w_width * self.w_height)
            y = location // (self.w_width)
            x = location%(self.w_width)

            beta[(2*(self.chebyshev_number**2)*dot_in_A_y)+((self.chebyshev_number**2)*dot_in_A_x) +
                 (self.chebyshev_number*chebs_height)+chebs_width][0][y][x] = \
                np.exp(-2*np.pi*1j *
                       ((xi_x[dot_in_A_x]*(-chebyshev_dots[chebs_width]+single_input_x[x])) +
                         (xi_y[dot_in_A_y]*(-chebyshev_dots[chebs_height]+single_input_y[y])))) * \
                Lagrange_Polynomial(single_input_x[x], chebyshev_dots, chebs_width) * \
                Lagrange_Polynomial(single_input_y[y], chebyshev_dots, chebs_height)


        '''Generate wt_fst_lyr_single'''
        for channel in range(4*self.chebyshev_number**2):
            wt_fst_lyr_single[4*channel, :1]    = beta[channel].real
            wt_fst_lyr_single[4*channel, 1:2]   = -beta[channel].imag
            wt_fst_lyr_single[4*channel, 2:3]   = -beta[channel].real
            wt_fst_lyr_single[4*channel, 3:]    = beta[channel].imag

            wt_fst_lyr_single[4*channel+1, :1]  = beta[channel].imag
            wt_fst_lyr_single[4*channel+1, 1:2] = beta[channel].real
            wt_fst_lyr_single[4*channel+1, 2:3] = -beta[channel].imag
            wt_fst_lyr_single[4*channel+1, 3:]  = -beta[channel].real

            wt_fst_lyr_single[4*channel+2] = -wt_fst_lyr_single[4*channel]
            wt_fst_lyr_single[4*channel+3] = -wt_fst_lyr_single[4*channel+1]
        return nn.Parameter(torch.tensor(wt_fst_lyr_single, dtype=torch.float32))



    def generate_Recursion_Layer_Weights(self, A_row_index_last_lyr, A_column_index_last_lyr, layer_num):
        xi_x                = np.array([self.left_frequency_kx+
                                        (A_row_index_last_lyr+0.25)*self.frequency_kx_length/(2**layer_num),
                                        self.left_frequency_kx+
                                        (A_row_index_last_lyr+0.75)*self.frequency_kx_length/(2**layer_num)])
        xi_y                = np.array([self.left_frequency_ky+
                                        (A_column_index_last_lyr+0.25)*self.frequency_ky_length/(2**layer_num),
                                        self.left_frequency_ky+
                                        (A_column_index_last_lyr+0.75)*self.frequency_ky_length/(2**layer_num)])
        chebyshev_dots      = Chebyshev_Nodes([0, 1 / (2 ** (self.layer_number - 1 - layer_num))],
                                              self.chebyshev_number)

        single_input_x      = Chebyshev_Nodes([0, 1 / (2 ** (self.layer_number - layer_num))], self.chebyshev_number) # from previous layer
        single_input_y      = Chebyshev_Nodes([0, 1 / (2 ** (self.layer_number - layer_num))], self.chebyshev_number) # from previous layer

        wt_rcs_lyr_single   = np.zeros((4*4*self.chebyshev_number**2,
                                        4*self.chebyshev_number**2,
                                        2,
                                        2))
        beta                = np.zeros((4*self.chebyshev_number**2,
                                        self.chebyshev_number**2,
                                        2,
                                        2), dtype=complex)

        '''Generate beta'''
        for dot in range(4*(self.chebyshev_number**4)*4):
            dot_in_A = dot//((self.chebyshev_number**4)*4)
            dot_in_A_y = dot_in_A//2
            dot_in_A_x = dot_in_A%2
            leftdots1 = dot%((self.chebyshev_number**4)*4)
            chebs = leftdots1//((self.chebyshev_number**2)*4)
            chebs_height = chebs//(self.chebyshev_number)
            chebs_width = chebs%(self.chebyshev_number)
            leftdots2 = leftdots1%((self.chebyshev_number**2)*4)
            predots = leftdots2//4
            y_dot = predots // (self.chebyshev_number)
            x_dot = predots % (self.chebyshev_number)
            location = leftdots2%4
            y = location // 2
            x = location % 2


            beta[(2*(self.chebyshev_number**2)*dot_in_A_y)+(self.chebyshev_number**2*dot_in_A_x)+
                 (self.chebyshev_number * chebs_height)+chebs_width][y_dot*self.chebyshev_number+x_dot][y][x] = \
                np.exp(-2 * np.pi * 1j *(
                       (xi_x[dot_in_A_x]*(-chebyshev_dots[chebs_width] +
                                          single_input_x[x_dot] +
                                          x/((2 ** (self.layer_number - layer_num))))) +
                       (xi_y[dot_in_A_y]*(-chebyshev_dots[chebs_height] +
                                          single_input_y[y_dot] +
                                          y/((2 ** (self.layer_number - layer_num))))))) * \
                Lagrange_Polynomial(single_input_x[x_dot] +
                                    x/((2 ** (self.layer_number - layer_num))),
                                    chebyshev_dots, chebs_width) * \
                Lagrange_Polynomial(single_input_y[y_dot] +
                                    y/((2 ** (self.layer_number - layer_num))),
                                    chebyshev_dots, chebs_height)
        '''Generate wt_rcs_lyr_single'''
        for out_channel in range(4*self.chebyshev_number**2):
            for in_channel in range(self.chebyshev_number**2):
                wt_rcs_lyr_single[4*out_channel, 4*in_channel:4*in_channel + 1]       = beta[out_channel, in_channel].real
                wt_rcs_lyr_single[4*out_channel, 4*in_channel + 1:4*in_channel + 2]   = -beta[out_channel, in_channel].imag
                wt_rcs_lyr_single[4*out_channel, 4*in_channel + 2:4*in_channel + 3]   = -beta[out_channel, in_channel].real
                wt_rcs_lyr_single[4*out_channel, 4*in_channel + 3:4*(in_channel+1)]   = beta[out_channel, in_channel].imag

                wt_rcs_lyr_single[4*out_channel+1, 4*in_channel:4*in_channel + 1]       = beta[out_channel, in_channel].imag
                wt_rcs_lyr_single[4*out_channel+1, 4*in_channel + 1:4*in_channel + 2]   = beta[out_channel, in_channel].real
                wt_rcs_lyr_single[4*out_channel+1, 4*in_channel + 2:4*in_channel + 3]   = -beta[out_channel, in_channel].imag
                wt_rcs_lyr_single[4*out_channel+1, 4*in_channel + 3:4*(in_channel+1)]   = -beta[out_channel, in_channel].real

            wt_rcs_lyr_single[4*out_channel+2] = -wt_rcs_lyr_single[4*out_channel]
            wt_rcs_lyr_single[4*out_channel+3] = -wt_rcs_lyr_single[4*out_channel+1]
        return nn.Parameter(torch.tensor(wt_rcs_lyr_single, dtype=torch.float32))

    def generate_FT_Layer_Weights(self, A_row_index_last_lyr, A_column_index_last_lyr):
        xi_x = [self.frequency_kx_uni_dots[self.leftover_kx*A_row_index_last_lyr+i] for i in range(self.leftover_kx)]
        xi_y = [self.frequency_ky_uni_dots[self.leftover_ky*A_column_index_last_lyr+i] for i in range(self.leftover_ky)]

        chebyshev_row_dots = Chebyshev_Nodes([0, 1], self.chebyshev_number)
        chebyshev_column_dots = Chebyshev_Nodes([0, 1], self.chebyshev_number)

        wt_ft_lyr_single = np.zeros((4*self.leftover_kx*self.leftover_ky,
                                     4 * self.chebyshev_number ** 2,
                                     1,
                                     1))
        alpha = np.zeros((self.leftover_kx*self.leftover_ky,
                          self.chebyshev_number ** 2,
                          1,
                          1), dtype=complex)

        '''Generate alpha'''
        for dots in range((self.chebyshev_number**2)*self.leftover_kx*self.leftover_ky):
            chebs = dots//(self.leftover_kx*self.leftover_ky)
            chebs_column = chebs//(self.chebyshev_number)
            chebs_row = chebs%(self.chebyshev_number)

            location = dots%(self.leftover_kx*self.leftover_ky)
            ky = location//(self.leftover_kx)
            kx = location%(self.leftover_kx)
            alpha[ky*self.leftover_kx+kx][chebs_column*self.chebyshev_number+chebs_row][0][0] = \
                np.exp((-2*np.pi*1j)*(xi_x[kx]*chebyshev_row_dots[chebs_row] +
                                                        xi_y[ky]*chebyshev_column_dots[chebs_column]))
        '''Generate wt_ft_lyr_single'''
        for out_channel in range(self.leftover_kx*self.leftover_ky):
            for in_channel in range(self.chebyshev_number**2):
                wt_ft_lyr_single[4*out_channel:4*out_channel+1, 4*in_channel:4*in_channel+1]       = alpha[out_channel, in_channel].real
                wt_ft_lyr_single[4*out_channel:4*out_channel+1, 4*in_channel+1:4*in_channel+2]     = -alpha[out_channel, in_channel].imag
                wt_ft_lyr_single[4*out_channel:4*out_channel+1, 4*in_channel+2:4*in_channel+3]     = -alpha[out_channel, in_channel].real
                wt_ft_lyr_single[4*out_channel:4*out_channel+1, 4*in_channel+3:4*(in_channel+1)]   = alpha[out_channel, in_channel].imag

                wt_ft_lyr_single[4*out_channel+1:4*out_channel+2, 4*in_channel:4*in_channel+1]      = alpha[out_channel, in_channel].imag
                wt_ft_lyr_single[4*out_channel+1:4*out_channel+2, 4*in_channel+1:4*in_channel+2]    = alpha[out_channel, in_channel].real
                wt_ft_lyr_single[4*out_channel+1:4*out_channel+2, 4*in_channel+2:4*in_channel+3]    = -alpha[out_channel, in_channel].imag
                wt_ft_lyr_single[4*out_channel+1:4*out_channel+2, 4*in_channel+3:4*(in_channel+1)]  = -alpha[out_channel, in_channel].real

            wt_ft_lyr_single[4*out_channel+2:4*out_channel+3] = -wt_ft_lyr_single[4*out_channel:4*out_channel+1]
            wt_ft_lyr_single[4*out_channel+3:4*out_channel+4] = -wt_ft_lyr_single[4*out_channel+1:4*out_channel+2]

        return nn.Parameter(torch.tensor(wt_ft_lyr_single, dtype=torch.float32))

    def joint(self, out):
        '''
        :param out: [batch_size, self.in_channel_num*4*4**(self.layer_number), 1, 1]
        :return:
        '''
        out_joint = torch.zeros(out.shape[0],
                                self.in_channel_num*self.frequency_kx_length*self.frequency_ky_length,
                                1,
                                1, dtype=torch.complex64)

        for channel in range(out_joint.shape[1]):
            out_joint[:, channel, 0, 0] = out[:, 4 * channel, 0, 0] + \
                                          1j * out[:, 4 * channel + 1, 0, 0] + \
                                          -out[:, 4 * channel + 2, 0, 0] + \
                                          -1j * out[:, 4 * channel + 3, 0, 0]
        return out_joint
