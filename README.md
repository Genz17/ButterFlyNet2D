# ButterFlyNet2D
This is a repository for ButterFlyNet2D.

## Funcs
Here stored some supportive functions which will be used in the building-up, training or testing process.

Before the training process, we highly recommend you to run ` BaseInitGenerate.py ` in this folder like this:

```
python BaseInitGenerate.py <input size> <layer number> <cheb num> <initMethod> <pretrain>    
```

This will do the initialization and save the parameters which can be reused in training.

## Nets

Core neural networks are stored here. You may see the structure in `Img/Example.svg`, which is an example of ButterFlyNet2D with 2 layers and 2 cheb num,  dealing with $4\times4$ input. The picture is
drawed through netron.

## Test
Mainly contain test parts. `Test.py` is implemented for testing, while other programs are for corresponding task's testing process. Test will be done during the training process. If you want to execute `Test.py`  only, run in this form:

```
python Test.py <task name> <dataset name> <image size> <local size> <net layer> <cheb num> <initMethod> <pretrain> <pic>
```

Here the option `<pic>` is for whether you want to save images.

## Train

Mainly contain training parts. 

If you want to do the training tasks, adjust settings in `settings.json`:

```
{
    "datasetName": "Celeba", 
    "task": "Inpaint", 
    "epoches": 12,
    "batch_size_train": 20,
    "image_size": 64,
    "local_size": 64,
    "net_layer": 6,
    "cheb_num": 2,
    "initMethod": "Fourier", 
    "pretrain" : true,
    "resume": false 
}
```

Then run `train.py`:

```
python train.py
```

If you want to do the Fourier Transform Approximation part, do
```
python FTApprox.py <test type> <corresponding number> <inputSize> Fourier <net type> <train>
```
here `<test type>` could be `layer` or `cheb`, `<correspoding number>` then become fixed cheb num or layer num. `<net type>` is designed for whether it is DFT or IDFT operator, could be `f` or `b`. `<train>` is to test
the approximation power before or after training, simply choose from `True` or `False`. After training, the parameters will be stored.
