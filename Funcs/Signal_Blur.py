from Gaussian_Func import gauss
import numpy as np


def signal_blur(real_signal, std, kernel_size):

    fq_gs = np.fft.fft2(gauss(0,0,std,kernel_size), real_signal.shape)
    fq_signal = np.fft.fft2(real_signal)
    fq_blur_signal = fq_gs * fq_signal
    blur_signal = np.fft.ifft2(fq_blur_signal).real

    return blur_signal
