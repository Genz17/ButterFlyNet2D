import torch
from inpaint_test_func      import test_inpainting
from deblur_test_func       import test_deblurring
from denoise_test_func      import test_denoising

def test(task,test_loader,batch_size_test,Net,image_size,local_size):
    with torch.no_grad():
        if task == 'Inpaint':
            test_inpainting(test_loader,batch_size_test,Net,image_size,local_size)
        elif task == 'Deblur':
            test_deblurring(test_loader, batch_size_test, Net, image_size)
        elif task == 'Denoise':
            test_denoising(test_loader, batch_size_test, Net, image_size, local_size)