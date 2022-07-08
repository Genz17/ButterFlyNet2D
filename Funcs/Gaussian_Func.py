import numpy as np
def gauss(mean_x, mean_y, std, out_size):
    func = lambda x, y: (1 / (2 * np.pi * (std ** 2))) * np.exp(
        -((x - mean_x) ** 2 + (y - mean_y) ** 2) / (2 * (std ** 2)))
    MAT = np.zeros(out_size)
    for row in range(MAT.shape[0]):
        for column in range(MAT.shape[1]):
            MAT[row][column] = func(row, column)
    total_value = sum(sum(MAT))
    MAT = MAT/total_value
    return MAT