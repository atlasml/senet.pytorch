dependencies = ["torch", "torchvision", "scipy"]

import torchvision.transforms as transforms
from sotabench.image_classification import imagenet, cifar10
import torch
import PIL

def se_resnet20(**kwargs):
    from senet.se_resnet import se_resnet20 as _se_resnet20

    return _se_resnet20(**kwargs)


def se_resnet56(**kwargs):
    from senet.se_resnet import se_resnet56 as _se_resnet56

    return _se_resnet56(**kwargs)


def se_resnet50(**kwargs):
    from senet.se_resnet import se_resnet50 as _se_resnet50

    return _se_resnet50(**kwargs)


def se_resnet101(**kwargs):
    from senet.se_resnet import se_resnet101 as _se_resnet101

    return _se_resnet101(**kwargs)

def benchmark():
    urllib.request.urlretrieve('https://www.dropbox.com/s/xpq8ne7rwa4kg4c/seresnet50-60a8950a85b2b.pkl', 'seresnet50-60a8950a85b2b.pkl')
    senet = se_resnet50(num_classes=1000)
    senet.load_state_dict(torch.load("seresnet50-60a8950a85b2b.pkl"))
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    input_transform = transforms.Compose([
        transforms.Resize(256, PIL.Image.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])
    
    imagenet.benchmark(
        model=hub_model,
        paper_model_name='SE-ResNet-50',
        paper_arxiv_id='1709.01507',
        paper_pwc_id='squeeze-and-excitation-networks',
        input_transform=input_transform,
        batch_size=256,
        num_gpu=1
    )
