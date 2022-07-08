import numpy as np
from Gaussian_Func import gauss

def signal(freq_x_size, freq_y_size, real_range_low, real_range_high, imag_range_low, imag_range_high, freq,
           input_x_size, input_y_size):
    MAT = np.random.uniform(real_range_low, real_range_high, (freq_x_size, freq_y_size)) + \
          1j * np.random.uniform(imag_range_low, imag_range_high, (freq_x_size, freq_y_size))

    MAT[0][0] = MAT[0][0].real
    if freq == 'low':
        conv_freq = MAT * gauss(0, 0, 1, (freq_x_size, freq_y_size))
        real_input = np.fft.ifft2(conv_freq, (input_x_size, input_y_size)).real
    else:
        conv_freq = MAT * gauss(8, 8, 1, (freq_x_size, freq_y_size))
        real_input = np.fft.ifft2(conv_freq, (input_x_size, input_y_size)).real
    return real_input