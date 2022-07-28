import matplotlib.pyplot as plt
from pylab import xticks
import numpy as np


def LossPlot(Xiter, Yiter, epochNum, path):
    fig, ax1 = plt.subplots()
    ax1.set_ylabel('loss') 
    ax1.set_xlabel('iter') 
    ax1.plot(Xiter, Yiter)
    ax1.tick_params(axis ='x', labelcolor = 'tab:blue')
    ax2 = ax1.twiny() 
    ax2.set_xlabel('epoch', color = 'tab:red') 
    xticks(range(epochNum))
    plt.savefig(path)
