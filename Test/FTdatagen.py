import numpy as np
np.random.seed(17)

train_batch = 10000
test_batch = 2000

size = 128 # size \times size

# train batch
dataTrain = np.random.random((train_batch,1,size,size))

# test batch
dataTest = np.random.random((test_batch,1,size,size))