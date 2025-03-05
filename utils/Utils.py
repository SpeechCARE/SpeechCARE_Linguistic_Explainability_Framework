import torch

def report(text, space = False):
    print(text)
    if space: print('-' * 50)

def free_gpu_memory():
    import gc
    gc.collect()
    torch.cuda.empty_cache()
