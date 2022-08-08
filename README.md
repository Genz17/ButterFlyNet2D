# ButterFlyNet2D
This is a repository for ButterFlyNet2D.

## Funcs
Here stored some supportive functions which will be used in the building-up, training or testing process.

Before the training process, we highly recommend you to run ``` BaseInitGenerate.py ``` in this folder like this:
```
python BaseInitGenerate.py <input size> <layer number> <cheb num> <prefix> <pretrain>    
```
This will do the initialization and save the parameters which can be reused in training.

We recommend you to run 
```
python BaseInitGenerate.py 64 6 2 True True
```
```
python BaseInitGenerate.py 32 5 2 True True
```
```
python BaseInitGenerate.py 16 4 2 True True
```
at first. These could be used in training.

## Nets

Core neural networks are stored here.

- **```ButterFlyNet2D.py```, ```ButterFlyNet2D_IDFT.py```**
These are a pair of neural networks that put the non-zero parameters in Butterfly Algorithm into CNN. ButterFlyNet2D is for DFT, meanwhile ButterFlyNet2D_IDFT is for IDFT.  

- **```ButterFlyNet2D_CNN.py```, ```ButterFlyNet2D_CNN_IDFT.py```**
These are a pair of neural networks that put all the parameters in Butterfly Algorithm into CNN. ButterFlyNet2D_CNN is for DFT, meanwhile ButterFlyNet2D_CNN_IDFT is for IDFT.

- **```ButterFlyNet2D_Flexible.py```, ```ButterFlyNet2D_Flexible_IDFT.py```**
These are a pair of neural networks that put the non-zero parameters in Butterfly Algorithm into CNN. ButterFlyNet2D is for DFT, meanwhile ButterFlyNet2D_IDFT is for IDFT. But the implementation make the stride from bigger than 1 into 1.

- **```ButterFlyNet2D_CNN_Flexible.py```**
This is a neural network that put all the parameters in Butterfly Algorithm into CNN. ButterFlyNet2D_CNN is for DFT. But the implementation make the stride from bigger than 1 into 1.

- **```ButterFlyNet2D_Identicle.py```**
  Assembles the ButterFlyNet2D and ButterFlyNet2D_IDFT into one whole neural network.


## Test
Mainly contain test parts. ```Test.py``` is implemented for testing, while other programs are for corresponding task's testing process. Test will be done during the training process. If you want to execute ```Test.py ``` only, run in this form:
```
python <task name> <dataset name> <image size> <local size> <prefix> <pretrain> <pic>
```
Here the option ```<pic>``` is for whether you want to save images.

## Train

Mainly contain training parts. 

Run ``` train.py ``` in such form:

```
python -u train.py <dataset name> <task name> <epoches> <batch size> <input size> <net input size> <net layer> <cheb num> <prefix> <pretrain> <resume>
```

For example:
```
python -u train.py Celeba Inpaint 12 12 128 64 5 2 True True False
```


