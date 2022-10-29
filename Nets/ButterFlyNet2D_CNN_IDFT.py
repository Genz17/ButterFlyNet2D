from Chebyshev_Nodes import *
from Lagrange_Interpolation import *
import torch
import numpy as np
import torch.nn as nn

class ButterFlyNet2D_CNN_IDFT(nn.Module):
    def __init__(self, in_channel,
                 left_frequency_kx, right_frequency_kx,
                 left_frequency_ky, right_frequency_ky,
                 height, width,
                 layer_number, chebyshev_number,prefixed,positivereal):
        super(ButterFlyNet2D_CNN_IDFT, self).__init__()

        self.in_channel_num         = in_channel
        self.left_frequency_kx      = left_frequency_kx
        self.right_frequency_kx     = right_frequency_kx
        self.left_frequency_ky      = left_frequency_ky
        self.right_frequency_ky     = right_frequency_ky
        self.frequency_kx_length    = right_frequency_kx - left_frequency_kx
        self.frequency_ky_length    = right_frequency_ky - left_frequency_ky
        self.layer_number           = layer_number
        self.w_ky                   = int(self.frequency_ky_length / (2 ** (self.layer_number - 1)))
        self.w_kx                   = int(self.frequency_kx_length / (2 ** (self.layer_number - 1)))
        self.chebyshev_number       = chebyshev_number
        self.height                 = height
        self.width                  = width
        self.leftover_y             = int(self.height / (2 ** layer_number))
        self.leftover_x             = int(self.width / (2 ** layer_number))
        self.x_uni_dots             = np.array(range(0, width)) / width
        self.y_uni_dots             = np.array(range(0, height)) / height
        self.plot_x, self.plot_y    = np.meshgrid(self.height, self.width)
        self.prefixed               = prefixed
        self.positivereal           = positivereal
        self.conv_dict              = self.Generate_Convs()


    def Generate_Convs(self):
        conv_dict_first = {str(0):
                               nn.Conv2d(in_channels=4*self.in_channel_num,
                                         out_channels=4 * 4 * (self.chebyshev_number ** 2)*self.in_channel_num,
                                         kernel_size=(self.w_ky, self.w_kx),
                                         stride=(self.w_ky, self.w_kx),
                                         bias=True)}
        conv_dict_rcs = {str(lyr):
                             nn.Conv2d(in_channels=4*(4**lyr) * (self.chebyshev_number ** 2)*self.in_channel_num,
                                       out_channels=4*(4**(lyr+1)) * (self.chebyshev_number ** 2)*self.in_channel_num,
                                       kernel_size=(2, 2),
                                       stride=(2, 2),
                                       bias=True)
                         for lyr in range(1, self.layer_number)}
        conv_dict_ft = {str(self.layer_number):
                            nn.Conv2d(in_channels=4*(4**self.layer_number) * (self.chebyshev_number ** 2)*self.in_channel_num,
                                      out_channels=4*self.height*self.width*self.in_channel_num,
                                      kernel_size=(1, 1),
                                      stride=(1, 1),
                                      bias=True)}

        conv_dict = {}
        conv_dict.update(conv_dict_first)
        conv_dict.update(conv_dict_rcs)
        conv_dict.update(conv_dict_ft)


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

            for key in conv_dict.keys():
                conv_dict[key].weight   = conv_Weights_dict[key]
                conv_dict[key].bias     = nn.Parameter(torch.zeros_like(conv_dict[key].bias))
        else:
            for key in conv_dict.keys():
                conv_dict[key].weight   = nn.init.normal_(conv_dict[key].weight, mean=0, std=1)
                conv_dict[key].bias     = nn.init.normal_(conv_dict[key].bias, mean=0, std=1)
        return nn.ModuleDict(conv_dict)

    def forward(self, input_data):
        out = self.split(input_data)
        out_rcs = self.conv_dict[str(0)](out)
        out_rcs = nn.ReLU(inplace=True)(out_rcs)

        for lyr in range(1, self.layer_number):
            out_rcs = self.conv_dict[str(lyr)](out_rcs)
            out_rcs = nn.ReLU(inplace=True)(out_rcs)

        out_final = self.conv_dict[str(self.layer_number)](out_rcs)
        out_final = nn.ReLU(inplace=True)(out_final)
        out_joint = self.joint(out_final)
        out_A = out_joint.view((input_data.shape[0], self.in_channel_num, self.height, self.width))/(self.height*self.width)


        # ax = plt.axes(projection='3d')
        # ax.scatter(self.plot_x, self.plot_y, out_A[0][0].real.detach().numpy())
        # ax.scatter(self.plot_x, self.plot_y, out_A[0][0].imag.detach().numpy())
        # plt.show()
        return out_A


    def split(self, input_data):
        data_split = torch.zeros((input_data.shape[0],
                     4*input_data.shape[1],
                     input_data.shape[2],
                     input_data.shape[3]))

        for channel in range(input_data.shape[1]):
            data_split[:, 4*channel] = \
                input_data[:, channel].real
            data_split[:, 4*channel+1] = \
                input_data[:, channel].imag
            data_split[:, 4*channel+2] = \
                -input_data[:, channel].real
            data_split[:, 4*channel+3] = \
                -input_data[:, channel].imag
        data_split = nn.ReLU(inplace=True)(data_split)
        return data_split

    def generate_First_Layer_Weights(self):
        # layer 0
        single_input_x = range(self.left_frequency_kx, self.right_frequency_kx)
        single_input_y = range(self.left_frequency_ky, self.right_frequency_ky)
        xi_x = np.array([1 / 4, 3 / 4])
        xi_y = np.array([1 / 4, 3 / 4])
        chebyshev_dots_kx = Chebyshev_Nodes([self.left_frequency_kx,
                                             self.left_frequency_kx + self.frequency_kx_length / (
                                                         2 ** (self.layer_number - 1))], self.chebyshev_number)
        chebyshev_dots_ky = Chebyshev_Nodes([self.left_frequency_ky,
                                             self.left_frequency_ky + self.frequency_ky_length / (
                                                         2 ** (self.layer_number - 1))], self.chebyshev_number)
        wt_fst_lyr_single       = np.zeros((4*4*(self.chebyshev_number**2)*self.in_channel_num,
                                            4*self.in_channel_num,
                                            self.w_ky,
                                            self.w_kx))

        beta                    = np.zeros((4*(self.chebyshev_number**2)*self.in_channel_num,
                                            self.in_channel_num,
                                            self.w_ky,
                                            self.w_kx), dtype=complex)


        '''Generate beta'''
        for channel in range(self.in_channel_num):
            for dot_in_A_y in range(2): # for the dots in A_y
                for dot_in_A_x in range(2): # for the dots in A_x
                    for chebs_height in range(self.chebyshev_number): # for the dots in B_y
                        for chebs_width in range(self.chebyshev_number): # for the dots in B_x
                            for y in range(self.w_ky):
                                for x in range(self.w_kx):
                                    beta[channel*4*(self.chebyshev_number**2)+
                                         (2*(self.chebyshev_number**2)*dot_in_A_y)+
                                         ((self.chebyshev_number**2)*dot_in_A_x) +
                                         (self.chebyshev_number*chebs_height)+chebs_width][channel][y][x] = \
                                        np.exp(2*np.pi*1j *
                                               ((xi_x[dot_in_A_x]*(-chebyshev_dots_kx[chebs_width]+single_input_x[x])) +
                                                 (xi_y[dot_in_A_y]*(-chebyshev_dots_ky[chebs_height]+single_input_y[y])))) * \
                                        Lagrange_Polynomial(single_input_x[x], chebyshev_dots_kx, chebs_width) * \
                                        Lagrange_Polynomial(single_input_y[y], chebyshev_dots_ky, chebs_height)


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
        return nn.Parameter(torch.FloatTensor(wt_fst_lyr_single))



    def generate_Recursion_Layer_Weights(self, layer_num):
        xi_x                = np.array([(2*i+1)/(2**(layer_num+2)) for i in range(2**(layer_num+1))])
        xi_y                = np.array([(2*i+1)/(2**(layer_num+2)) for i in range(2**(layer_num+1))])
        chebyshev_dots_kx = Chebyshev_Nodes([self.left_frequency_kx,
                                             self.left_frequency_kx + self.frequency_kx_length / (
                                                         2 ** (self.layer_number - 1 - layer_num))],
                                            self.chebyshev_number)
        chebyshev_dots_ky = Chebyshev_Nodes([self.left_frequency_ky,
                                             self.left_frequency_ky + self.frequency_ky_length / (
                                                         2 ** (self.layer_number - 1 - layer_num))],
                                            self.chebyshev_number)

        single_input_x = Chebyshev_Nodes([self.left_frequency_kx, self.left_frequency_kx + self.frequency_kx_length / (
                    2 ** (self.layer_number - layer_num))], self.chebyshev_number)  # from previous layer
        single_input_y = Chebyshev_Nodes([self.left_frequency_ky, self.left_frequency_ky + self.frequency_ky_length / (
                    2 ** (self.layer_number - layer_num))], self.chebyshev_number)

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
                                                        np.exp(2 * np.pi * 1j *(
                                                               (xi_x[2*Ax+dot_in_A_x]*(-chebyshev_dots_kx[chebs_width] +
                                                                                  single_input_x[x_dot] +
                                                                                  x*self.frequency_kx_length/((2 ** (self.layer_number - layer_num))))) +
                                                               (xi_y[2*Ay+dot_in_A_y]*(-chebyshev_dots_ky[chebs_height] +
                                                                                  single_input_y[y_dot] +
                                                                                  y*self.frequency_ky_length/((2 ** (self.layer_number - layer_num))))))) * \
                                                        Lagrange_Polynomial(single_input_x[x_dot] +
                                                                            x*self.frequency_kx_length/((2 ** (self.layer_number - layer_num))),
                                                                            chebyshev_dots_kx, chebs_width) * \
                                                        Lagrange_Polynomial(single_input_y[y_dot] +
                                                                            y*self.frequency_ky_length/((2 ** (self.layer_number - layer_num))),
                                                                            chebyshev_dots_ky, chebs_height)

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
        return nn.Parameter(torch.FloatTensor(wt_rcs_lyr_single))

    def generate_FT_Layer_Weights(self):
        chebyshev_row_dots = Chebyshev_Nodes([self.left_frequency_kx, self.right_frequency_kx], self.chebyshev_number)
        chebyshev_column_dots = Chebyshev_Nodes([self.left_frequency_ky, self.right_frequency_ky], self.chebyshev_number)

        wt_ft_lyr_single = np.zeros((4*self.height*self.width*self.in_channel_num,
                                     4 * (4**self.layer_number) * (self.chebyshev_number ** 2)*self.in_channel_num,
                                     1,
                                     1))
        alpha = np.zeros((self.height*self.width*self.in_channel_num,
                          (4**self.layer_number) * (self.chebyshev_number ** 2)*self.in_channel_num,
                          1,
                          1), dtype=complex)

        '''Generate alpha'''
        for channel in range(self.in_channel_num):
            for Ay in range(2**self.layer_number):
                for Ax in range(2**self.layer_number):
                    for chebs_column in range(self.chebyshev_number):  # for the dots in B_y
                        for chebs_row in range(self.chebyshev_number):  # for the dots in B_x
                            for ky in range(self.leftover_y):
                                for kx in range(self.leftover_x):
                                    alpha[channel*self.width*self.height+
                                          Ay*(2**self.layer_number)*self.leftover_y*self.leftover_x+
                                          Ax*self.leftover_x+
                                          ky*self.width+kx]\
                                    [channel*(4**self.layer_number) * (self.chebyshev_number ** 2)+
                                     Ay*(2**self.layer_number)*(self.chebyshev_number**2)+
                                     Ax*(self.chebyshev_number**2)+
                                     chebs_column*self.chebyshev_number+chebs_row][0][0] = \
                                        np.exp((2*np.pi*1j)*
                                               (self.x_uni_dots[Ax*self.leftover_x+kx]*chebyshev_row_dots[chebs_row]+
                                                self.y_uni_dots[Ay*self.leftover_y+ky]*chebyshev_column_dots[chebs_column]))

        '''Generate wt_ft_lyr_single'''
        for out_channel in range(self.height*self.width*self.in_channel_num):
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

        return nn.Parameter(torch.FloatTensor(wt_ft_lyr_single))

    def joint(self, out):
        '''
        :param out: [batch_size, self.in_channel_num*4*4**(self.layer_number), 1, 1]
        :return:
        '''
        if self.positivereal:
            out_joint = torch.zeros(out.shape[0],
                                    self.in_channel_num * self.height * self.width,
                                    1,
                                    1)
            for channel in range(out_joint.shape[1]):
                out_joint[:, channel, 0, 0] = out[:, 4 * channel, 0, 0] + \
                                              -out[:, 4 * channel + 2, 0, 0]
        else:
            out_joint = torch.zeros(out.shape[0],
                                    self.in_channel_num * self.height * self.width,
                                    1,
                                    1, dtype=torch.complex64)
            for channel in range(out_joint.shape[1]):
                out_joint[:, channel, 0, 0] = out[:, 4 * channel, 0, 0] + \
                                              1j * out[:, 4 * channel + 1, 0, 0] + \
                                              -out[:, 4 * channel + 2, 0, 0] + \
                                              -1j * out[:, 4 * channel + 3, 0, 0]
        return out_joint