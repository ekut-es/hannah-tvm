import torch 
import torchvision.datasets as datasets
import torchvision.transforms as transforms


def imagenet(batch_size=1):
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    dataset = datasets.ImageNet("datasets/imagenet", train=False, download=True, transform=preprocess)
    data_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=batch_size,
                                              shuffle=True,
                                              num_workers=4)

    return data_loader