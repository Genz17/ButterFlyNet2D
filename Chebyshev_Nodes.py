import numpy as np
import math

def Chebyshev_Nodes(dom, r):
    '''
    This is for getting r Chebyshev Nodes in area dom
    :param dom: 1x2 vector, [a, b], a < b
    :param r: number related to Chebyshev_Nodes
    :return: np.array of the size r
    '''
    w_x = dom[1] - dom[0]
    center_x = (w_x / 2) + dom[0]
    Chebyshev_Grid = np.array([w_x*0.5*np.cos((2*t+1)*math.pi/(2*r+2))+center_x for t in range(1, r+1, 1)])
    nodes = np.array(Chebyshev_Grid)

    return nodes