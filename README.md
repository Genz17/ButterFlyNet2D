# ButterFlyNet2D
This is a repository for ButterFlyNet2D.

## Nets

Core neural networks are stored here.

- **ButterFlyNet2D, ButterFlyNet2D_IDFT**
These are a pair of neural networks that put the non-zero parameters in Butterfly Algorithm into CNN. ButterFlyNet2D is for DFT, meanwhile ButterFlyNet2D_IDFT is for IDFT.  

- **ButterFlyNet2D_CNN, ButterFlyNet2D_CNN_IDFT**
These are a pair of neural networks that put all the parameters in Butterfly Algorithm into CNN. ButterFlyNet2D_CNN is for DFT, meanwhile ButterFlyNet2D_CNN_IDFT is for IDFT.

- **ButterFlyNet2D_Flexible, ButterFlyNet2D_Flexible_IDFT**
These are a pair of neural networks that put the non-zero parameters in Butterfly Algorithm into CNN. ButterFlyNet2D is for DFT, meanwhile ButterFlyNet2D_IDFT is for IDFT. But the implementation make the stride from bigger than 1 into 1.

- **ButterFlyNet2D_CNN_Flexible**
This is a neural network that put all the parameters in Butterfly Algorithm into CNN. ButterFlyNet2D_CNN is for DFT. But the implementation make the stride from bigger than 1 into 1.

- **ButterFlyNet2D_Identicle**
  Assembles the ButterFlyNet2D and ButterFlyNet2D_IDFT into one whole neural network.

## Train
Mainly contain training parts. Each file is for its corresponding task's training process.

## Test
Mainly contain test parts. **TEST.py** is implemented for testing, while other programs are for corresponding task's testing process.

## Funcs
Here stored some supportive functions which will be used in the building-up, training or testing process.
