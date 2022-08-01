# ButterFlyNet2D
This is a repository for ButterFlyNet2D.

## Nets

Core neural networks are stored here.

- **ButterFlyNet2D.py, ButterFlyNet2D_IDFT.py**
These are a pair of neural networks that put the non-zero parameters in Butterfly Algorithm into CNN. ButterFlyNet2D is for DFT, meanwhile ButterFlyNet2D_IDFT is for IDFT.  

- **ButterFlyNet2D_CNN.py, ButterFlyNet2D_CNN_IDFT.py**
These are a pair of neural networks that put all the parameters in Butterfly Algorithm into CNN. ButterFlyNet2D_CNN is for DFT, meanwhile ButterFlyNet2D_CNN_IDFT is for IDFT.

- **ButterFlyNet2D_Flexible.py, ButterFlyNet2D_Flexible_IDFT.py**
These are a pair of neural networks that put the non-zero parameters in Butterfly Algorithm into CNN. ButterFlyNet2D is for DFT, meanwhile ButterFlyNet2D_IDFT is for IDFT. But the implementation make the stride from bigger than 1 into 1.

- **ButterFlyNet2D_CNN_Flexible.py**
This is a neural network that put all the parameters in Butterfly Algorithm into CNN. ButterFlyNet2D_CNN is for DFT. But the implementation make the stride from bigger than 1 into 1.

- **ButterFlyNet2D_Identicle.py**
  Assembles the ButterFlyNet2D and ButterFlyNet2D_IDFT into one whole neural network.

## Train
Mainly contain training parts. Run train.py in such form:

```
python -u train.py <dataset name> <task name> <epoches> <batch size> <input size> <net input size> <prefix> <pretrain> <resume>
```

For example:
```
python -u train.py Celeba Inpaint 12 12 128 64 True True False
```

## Test
Mainly contain test parts. **TEST.py** is implemented for testing, while other programs are for corresponding task's testing process.

## Funcs
Here stored some supportive functions which will be used in the building-up, training or testing process.
