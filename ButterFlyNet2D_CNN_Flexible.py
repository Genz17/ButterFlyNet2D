from Chebyshev_Nodes import *
from Lagrange_Interpolation import *
from Period_Padding import period_padding
import torch
import numpy as np
import torch.nn as nn

class ButterFlyNet2D_CNN_Flexible(nn.Module):
    def __init__(self, in_channel, height, width,
                 layer_number, chebyshev_number,
                 left_frequency_kx, right_frequency_kx,
                 left_frequency_ky, right_frequency_ky, prefixed, stride_list, pooling_list,image,positivereal):
        super(ButterFlyNet2D_CNN_Flexible, self).__init__()

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
        self.leftover_kx            = int((right_frequency_kx - left_frequency_kx) / (2 ** layer_number))
        self.leftover_ky            = int((right_frequency_ky - left_frequency_ky) / (2 ** layer_number))
        self.frequency_kx_uni_dots  = np.array(range(left_frequency_kx, right_frequency_kx))
        self.frequency_ky_uni_dots  = np.array(range(left_frequency_ky, right_frequency_ky))
        self.plot_x, self.plot_y    = np.meshgrid(self.frequency_kx_uni_dots, self.frequency_ky_uni_dots)
        self.stride_list            = stride_list
        self.pooling_list           = pooling_list
        self.prefixed               = prefixed
        self.image                  = image
        self.positivereal           = positivereal
        self.conv_dict              = self.Generate_Convs()


    def Generate_Convs(self):
        if self.prefixed:
            conv_Weights_dict_first = {str(0):
                                           self.generate_First_Layer_Weights()}
            conv_Weights_dict_rcs = {str(lyr):
                                         self.generate_Recursion_Layer_Weights(lyr)
                                     for lyr in range(1, self.layer_number)}
            conv_Weights_dict_ft = {str(self.layer_number):
                                        self.generate_FT_Layer_Weights()}
            conv_Weights_dict = {}
            conv_Weights_dict.update(conv_Weights_dict_first)
            conv_Weights_dict.update(conv_Weights_dict_rcs)
            conv_Weights_dict.update(conv_Weights_dict_ft)

        conv_dict_first = {str(0):
                               nn.Conv2d(in_channels=4*self.in_channel_num,
                                         out_channels=4 * 4 * (self.chebyshev_number ** 2)*self.in_channel_num,
                                         kernel_size=(self.w_height, self.w_width),
                                         stride=(self.stride_list[0][0], self.stride_list[0][1]),
                                         bias=True).cuda()}
        conv_dict_rcs = {str(lyr):
                             nn.Conv2d(in_channels=4*(4**lyr) * (self.chebyshev_number ** 2)*self.in_channel_num,
                                       out_channels=4*(4**(lyr+1)) * (self.chebyshev_number ** 2)*self.in_channel_num,
                                       kernel_size=(2, 2),
                                       stride=(self.stride_list[lyr][0],self.stride_list[lyr][1]),
                                       bias=True).cuda()
                         for lyr in range(1, self.layer_number)}
        conv_dict_ft = {str(self.layer_number):
                            nn.Conv2d(in_channels=4*(4**self.layer_number) * (self.chebyshev_number ** 2)*self.in_channel_num,
                                      out_channels=4*self.frequency_kx_length*self.frequency_ky_length*self.in_channel_num,
                                      kernel_size=(1, 1),
                                      stride=(1, 1),
                                      bias=True).cuda()}

        conv_dict = {}
        conv_dict.update(conv_dict_first)
        conv_dict.update(conv_dict_rcs)
        conv_dict.update(conv_dict_ft)

        if self.image:
            linear = {
                str(self.layer_number + 1): nn.Linear(4 * self.in_channel_num * self.frequency_kx_length * self.frequency_ky_length,
                                                      10).cuda()}
            conv_dict.update(linear)

            nn.init.kaiming_normal_(conv_dict[str(self.layer_number + 1)].weight)
            conv_dict[str(self.layer_number + 1)].bias = nn.Parameter(
                torch.zeros_like(conv_dict[str(self.layer_number + 1)].bias))

        if self.pooling_list[0] == 'Adjust':
            AdjustConv_first = {str(('Adjust', 0)):
                                    nn.Conv2d(in_channels=1,
                                              out_channels=1,
                                              kernel_size=(self.w_height//self.stride_list[0][0],
                                                           self.w_width//self.stride_list[0][1]),
                                              stride=(self.w_height//self.stride_list[0][0],
                                                        self.w_width//self.stride_list[0][1]),
                                              bias=True).cuda()}
            conv_Weights_adjust_first = {str(('Adjust', 0)):
                                             self.generate_AdjustConv_Weights(0)}
            conv_dict.update(AdjustConv_first)
            conv_Weights_dict.update(conv_Weights_adjust_first)

        AdjustConv_rcs = {str(('Adjust', lyr)):
                              nn.Conv2d(in_channels=1,
                                        out_channels=1,
                                        kernel_size=(2 // self.stride_list[lyr][0], 2 // self.stride_list[lyr][1]),
                                        stride=(2 // self.stride_list[lyr][0], 2 // self.stride_list[lyr][1]),
                                        bias=True).cuda()
                          for lyr in range(1, self.layer_number) if self.pooling_list[lyr]=='Adjust'}
        conv_Weights_adjust_rcs = {str(('Adjust', lyr)):
                                       self.generate_AdjustConv_Weights(lyr)
                                   for lyr in range(1, self.layer_number) if self.pooling_list[lyr] == 'Adjust'}
        conv_dict.update(AdjustConv_rcs)
        conv_Weights_dict.update(conv_Weights_adjust_rcs)


        if self.prefixed:
            for key in conv_Weights_dict.keys():
                conv_dict[key].weight   = conv_Weights_dict[key]
                conv_dict[key].bias     = nn.Parameter(torch.zeros_like(conv_dict[key].bias, dtype=torch.float32))
        else:
            for key in conv_dict.keys():
                conv_dict[key].weight   = nn.init.normal_(conv_dict[key].weight, mean=0, std=1)
                conv_dict[key].bias     = nn.init.normal_(conv_dict[key].bias, mean=0, std=1)
        return nn.ModuleDict(conv_dict)

    def forward(self, input_data):
        out = self.split(input_data).cuda()
        out_rcs = self.conv_dict[str(0)](out)
        out_rcs = nn.ReLU(inplace=True)(out_rcs)
        out_pad = period_padding(out_rcs, self.w_width//self.stride_list[0][1]-1,self.w_height//self.stride_list[0][0]-1)
        if self.pooling_list[0] == 'Max':
            out_rcs = nn.MaxPool2d(kernel_size=(self.w_height//self.stride_list[0][0],
                                                           self.w_width//self.stride_list[0][1]),
                                   stride=(self.w_height//self.stride_list[0][0],
                                                           self.w_width//self.stride_list[0][1]))(out_pad)
        elif self.pooling_list[0] == 'Avg':
            out_rcs = nn.AvgPool2d(kernel_size=(self.w_height//self.stride_list[0][0],
                                                           self.w_width//self.stride_list[0][1]),
                                   stride=(self.w_height//self.stride_list[0][0],
                                                           self.w_width//self.stride_list[0][1]))(out_pad)
        else:
            out_rcs = torch.zeros(out_pad.shape[0],
                                  out_pad.shape[1],
                                    2 ** (self.layer_number - 1),
                                    2 ** (self.layer_number - 1), dtype=torch.float32, device='cuda:0')
            for inchannel in range(out_rcs.shape[1]):
                out_rcs[:,inchannel:inchannel+1] = self.conv_dict[str(('Adjust', 0))](out_pad[:,inchannel:inchannel+1])

        for lyr in range(1, self.layer_number):
            out_rcs = self.conv_dict[str(lyr)](out_rcs)
            out_rcs = nn.ReLU(inplace=True)(out_rcs)
            out_pad = period_padding(out_rcs, 2//self.stride_list[lyr][1]-1,2//self.stride_list[lyr][0]-1)
            if self.pooling_list[lyr] == 'Max':
                out_rcs = nn.MaxPool2d(kernel_size=(2//self.stride_list[lyr][0], 2//self.stride_list[lyr][1]),
                                       stride=(2//self.stride_list[lyr][0], 2//self.stride_list[lyr][1]))(out_pad)
            elif self.pooling_list[lyr] == 'Avg':
                out_rcs = nn.AvgPool2d(kernel_size=(2//self.stride_list[lyr][0], 2//self.stride_list[lyr][1]),
                                       stride=(2//self.stride_list[lyr][0], 2//self.stride_list[lyr][1]))(out_pad)
            else:
                out_rcs = torch.zeros(out_pad.shape[0],
                                      out_pad.shape[1],
                                      2 ** (self.layer_number - lyr - 1),
                                      2 ** (self.layer_number - lyr - 1), dtype=torch.float32, device='cuda:0')
                for inchannel in range(out_rcs.shape[1]):
                    out_rcs[:, inchannel:inchannel + 1] = self.conv_dict[str(('Adjust', lyr))](out_pad[:, inchannel:inchannel + 1])

        out_final = self.conv_dict[str(self.layer_number)](out_rcs)
        out_final = nn.ReLU(inplace=True)(out_final)
        if self.image:
            out_A = out_final.view((input_data.shape[0], -1))
            output = self.conv_dict[str(self.layer_number + 1)](out_A)
        else:
            out_joint = self.joint(out_final)
            output = out_joint.view((input_data.shape[0], self.in_channel_num, self.frequency_kx_length, self.frequency_ky_length))

        return output


    def split(self, input_data):
        data_split = torch.zeros((input_data.shape[0],
                     4*input_data.shape[1],
                     input_data.shape[2],
                     input_data.shape[3]))

        if self.positivereal:
            for channel in range(input_data.shape[1]):
                data_split[:, 4 * channel, :, :] = \
                    input_data[:, channel, :, :]
            data_split = nn.ReLU(inplace=True)(data_split)
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
            data_split = nn.ReLU(inplace=True)(data_split)
        return data_split

    def generate_First_Layer_Weights(self):
        # layer 0
        single_input_x          = np.linspace(0, (1-1/self.width), self.width)
        single_input_y          = np.linspace(0, (1-1/self.height), self.height)
        xi_x = np.array([self.left_frequency_kx + (self.frequency_kx_length / 4),
                         self.left_frequency_kx + (self.frequency_kx_length * 3 / 4)])
        xi_y = np.array([self.left_frequency_ky + (self.frequency_ky_length / 4),
                         self.left_frequency_ky + (self.frequency_ky_length * 3 / 4)])
        chebyshev_dots          = Chebyshev_Nodes([0, 1/(2**(self.layer_number-1))], self.chebyshev_number)
        wt_fst_lyr_single       = np.zeros((4*4*(self.chebyshev_number**2)*self.in_channel_num,
                                            4*self.in_channel_num,
                                            self.w_height,
                                            self.w_width))

        beta                    = np.zeros((4*(self.chebyshev_number**2)*self.in_channel_num,
                                            self.in_channel_num,
                                            self.w_height,
                                            self.w_width), dtype=complex)


        '''Generate beta'''
        for channel in range(self.in_channel_num):
            for dot_in_A_y in range(2): # for the dots in A_y
                for dot_in_A_x in range(2): # for the dots in A_x
                    for chebs_height in range(self.chebyshev_number): # for the dots in B_y
                        for chebs_width in range(self.chebyshev_number): # for the dots in B_x
                            for y in range(self.w_height):
                                for x in range(self.w_width):
                                    beta[channel*4*(self.chebyshev_number**2)+
                                         (2*(self.chebyshev_number**2)*dot_in_A_y)+
                                         ((self.chebyshev_number**2)*dot_in_A_x) +
                                         (self.chebyshev_number*chebs_height)+chebs_width][channel][y][x] = \
                                        np.exp(-2*np.pi*1j *
                                               ((xi_x[dot_in_A_x]*(-chebyshev_dots[chebs_width]+single_input_x[x])) +
                                                 (xi_y[dot_in_A_y]*(-chebyshev_dots[chebs_height]+single_input_y[y])))) * \
                                        Lagrange_Polynomial(single_input_x[x], chebyshev_dots, chebs_width) * \
                                        Lagrange_Polynomial(single_input_y[y], chebyshev_dots, chebs_height)


        '''Generate wt_fst_lyr_single'''
        for out_channel in range(4*(self.chebyshev_number**2)*self.in_channel_num):
            for in_channel in range(self.in_channel_num):
                wt_fst_lyr_single[4*out_channel, 4*in_channel:4*in_channel+1]    = beta[out_channel][in_channel].real
                wt_fst_lyr_single[4*out_channel, 4*in_channel+1:4*in_channel+2]   = -beta[out_channel][in_channel].imag
                wt_fst_lyr_single[4*out_channel, 4*in_channel+2:4*in_channel+3]   = -beta[out_channel][in_channel].real
                wt_fst_lyr_single[4*out_channel, 4*in_channel+3:4*(in_channel+1)]    = beta[out_channel][in_channel].imag

                wt_fst_lyr_single[4*out_channel+1, 4*in_channel:4*in_channel+1]  = beta[out_channel][in_channel].imag
                wt_fst_lyr_single[4*out_channel+1, 4*in_channel+1:4*in_channel+2] = beta[out_channel][in_channel].real
                wt_fst_lyr_single[4*out_channel+1, 4*in_channel+2:4*in_channel+3] = -beta[out_channel][in_channel].imag
                wt_fst_lyr_single[4*out_channel+1, 4*in_channel+3:4*(in_channel+1)] = -beta[out_channel][in_channel].real

            wt_fst_lyr_single[4*out_channel+2] = -wt_fst_lyr_single[4*out_channel]
            wt_fst_lyr_single[4*out_channel+3] = -wt_fst_lyr_single[4*out_channel+1]
        return nn.Parameter(torch.FloatTensor(wt_fst_lyr_single).cuda())



    def generate_Recursion_Layer_Weights(self, layer_num):
        xi_x                = np.array([self.left_frequency_kx+
                                        (2*i+1)*self.frequency_kx_length/(2**(layer_num+2))
                                        for i in range(2**(layer_num+1))])
        xi_y                = np.array([self.left_frequency_ky+
                                        (2*i+1)*self.frequency_ky_length/(2**(layer_num+2))
                                        for i in range(2**(layer_num+1))])
        chebyshev_dots      = Chebyshev_Nodes([0, 1 / (2 ** (self.layer_number - 1 - layer_num))],
                                              self.chebyshev_number)

        single_input_x      = Chebyshev_Nodes([0, 1 / (2 ** (self.layer_number - layer_num))], self.chebyshev_number) # from previous layer
        single_input_y      = Chebyshev_Nodes([0, 1 / (2 ** (self.layer_number - layer_num))], self.chebyshev_number) # from previous layer

        wt_rcs_lyr_single   = np.zeros((4*(4**(layer_num+1))*(self.chebyshev_number**2)*self.in_channel_num,
                                        4*(4**layer_num)*(self.chebyshev_number**2)*self.in_channel_num,
                                        2,
                                        2))
        beta                = np.zeros(((4**(layer_num+1))*(self.chebyshev_number**2)*self.in_channel_num,
                                        (4**layer_num)*(self.chebyshev_number**2)*self.in_channel_num,
                                        2,
                                        2), dtype=complex)

        '''Generate beta'''
        for channel in range(self.in_channel_num):
            for Ay in range(2**layer_num):
                for Ax in range(2**layer_num):
                    for dot_in_A_y in range(2):  # for the dots in A_y
                        for dot_in_A_x in range(2):  # for the dots in A_x
                            for chebs_height in range(self.chebyshev_number):  # for the dots in B_y, this layer
                                for chebs_width in range(self.chebyshev_number):  # for the dots in B_x, this layer
                                    for y_dot in range(self.chebyshev_number):  # these are the dots of last layer
                                        for x_dot in range(self.chebyshev_number):  # these are the dots of last layer
                                            for y in range(2):
                                                for x in range(2):
                                                    beta[channel*(4**(layer_num+1))*self.chebyshev_number**2+
                                                         Ay*(2**layer_num)*4*(self.chebyshev_number**2)+
                                                         Ax*2*(self.chebyshev_number**2)+
                                                         (self.chebyshev_number**2)*(2**layer_num)*2*dot_in_A_y+
                                                         (self.chebyshev_number**2)*dot_in_A_x+
                                                         (self.chebyshev_number * chebs_height)+chebs_width]\
                                                        [channel*(4**layer_num)*(self.chebyshev_number**2)+
                                                         Ay*(2**layer_num)*(self.chebyshev_number**2)+
                                                         Ax*(self.chebyshev_number**2)+
                                                         y_dot*self.chebyshev_number+x_dot][y][x] = \
                                                        np.exp(-2 * np.pi * 1j *(
                                                               (xi_x[2*Ax+dot_in_A_x]*(-chebyshev_dots[chebs_width] +
                                                                                  single_input_x[x_dot] +
                                                                                  x/((2 ** (self.layer_number - layer_num))))) +
                                                               (xi_y[2*Ay+dot_in_A_y]*(-chebyshev_dots[chebs_height] +
                                                                                  single_input_y[y_dot] +
                                                                                  y/((2 ** (self.layer_number - layer_num))))))) * \
                                                        Lagrange_Polynomial(single_input_x[x_dot] +
                                                                            x/((2 ** (self.layer_number - layer_num))),
                                                                            chebyshev_dots, chebs_width) * \
                                                        Lagrange_Polynomial(single_input_y[y_dot] +
                                                                            y/((2 ** (self.layer_number - layer_num))),
                                                                            chebyshev_dots, chebs_height)

        for out_channel in range((4**(layer_num+1))*(self.chebyshev_number**2)*self.in_channel_num):
            for in_channel in range((4**layer_num)*(self.chebyshev_number**2)*self.in_channel_num):
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
        return nn.Parameter(torch.FloatTensor(wt_rcs_lyr_single).cuda())

    def generate_FT_Layer_Weights(self):
        chebyshev_row_dots = Chebyshev_Nodes([0, 1], self.chebyshev_number)
        chebyshev_column_dots = Chebyshev_Nodes([0, 1], self.chebyshev_number)

        wt_ft_lyr_single = np.zeros((4*self.frequency_kx_length*self.frequency_ky_length*self.in_channel_num,
                                     4 * (4**self.layer_number) * (self.chebyshev_number ** 2)*self.in_channel_num,
                                     1,
                                     1))
        alpha = np.zeros((self.frequency_kx_length*self.frequency_ky_length*self.in_channel_num,
                          (4**self.layer_number) * (self.chebyshev_number ** 2)*self.in_channel_num,
                          1,
                          1), dtype=complex)

        '''Generate alpha'''
        for channel in range(self.in_channel_num):
            for Ay in range(2**self.layer_number):
                for Ax in range(2**self.layer_number):
                    for chebs_column in range(self.chebyshev_number):  # for the dots in B_y
                        for chebs_row in range(self.chebyshev_number):  # for the dots in B_x
                            for ky in range(self.leftover_ky):
                                for kx in range(self.leftover_kx):
                                    alpha[channel*self.frequency_ky_length*self.frequency_kx_length+
                                          Ay*(2**self.layer_number)*self.leftover_ky*self.leftover_kx+
                                          Ax*self.leftover_kx+
                                          ky*self.frequency_kx_length+kx]\
                                    [channel*(4**self.layer_number) * (self.chebyshev_number ** 2)+
                                     Ay*(2**self.layer_number)*(self.chebyshev_number**2)+
                                     Ax*(self.chebyshev_number**2)+
                                     chebs_column*self.chebyshev_number+chebs_row][0][0] = \
                                        np.exp((-2*np.pi*1j)*
                                               (self.frequency_kx_uni_dots[Ax*self.leftover_kx+kx]*chebyshev_row_dots[chebs_row]+
                                                self.frequency_ky_uni_dots[Ay*self.leftover_ky+ky]*chebyshev_column_dots[chebs_column]))

        '''Generate wt_ft_lyr_single'''
        for out_channel in range(self.frequency_kx_length*self.frequency_ky_length*self.in_channel_num):
            for in_channel in range((4**self.layer_number) * (self.chebyshev_number ** 2)*self.in_channel_num):
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

        return nn.Parameter(torch.FloatTensor(wt_ft_lyr_single).cuda())

    def joint(self, out):
        '''
        :param out: [batch_size, self.in_channel_num*4*4**(self.layer_number), 1, 1]
        :return:
        '''
        out_joint = torch.zeros(out.shape[0],
                                self.in_channel_num*self.frequency_kx_length*self.frequency_ky_length,
                                1,
                                1, dtype=torch.complex64).cuda()

        for channel in range(out_joint.shape[1]):
            out_joint[:, channel, 0, 0] = out[:, 4 * channel, 0, 0] + \
                                        1j*out[:, 4 * channel + 1, 0, 0] + \
                                        -out[:, 4 * channel + 2, 0, 0] + \
                                        -1j*out[:, 4 * channel + 3, 0, 0]
        return out_joint

    def generate_AdjustConv_Weights(self, lyr):
        if lyr == 0:
            wt_adjust = np.zeros((1,
                                  1,
                                  self.w_height // self.stride_list[0][0],
                                  self.w_width // self.stride_list[0][1]))
            wt_adjust[0][0][0][0] = 1

        else:
            wt_adjust = np.zeros((1,
                                  1,
                                  2 // self.stride_list[lyr][0],
                                  2 // self.stride_list[lyr][1]))
            wt_adjust[0][0][0][0] = 1

        return nn.Parameter(torch.tensor(wt_adjust, dtype=torch.float32, device='cuda:0'))