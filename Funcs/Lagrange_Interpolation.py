def Lagrange_Polynomial(x, Nodes, j):
    '''
    This is for getting L(x)
    :param x: L(x)
    :param Nodes: Interpolation Nodes, a list or an array
    :return: [\Pi_{i != j}(x - t_i)/(t_j - t_i) for j in range(len(Nodes))]
    '''

    poly = 1
    for i in range(len(Nodes)):
        if i != j:
            poly = poly * (x - Nodes[i]) / (Nodes[j] - Nodes[i])

    return poly