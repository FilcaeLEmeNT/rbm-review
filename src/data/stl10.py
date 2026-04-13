import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_stl10_loaders(batch_size=64, path='./data', verbose=False):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1)),
    ])

    train_data = datasets.STL10(
        root=path,
        train=True,
        transform=transform,
        download=True
    )

    test_data = datasets.STL10(
        root=path,
        train=False,
        transform=transform,
        download=True
    )

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    
    if verbose:
        print(f"Train dataset size: {len(train_data)}")
        print(f"Test dataset size: {len(test_data)}")
        print(f"Batch size: {batch_size}")
        print(f"Train batches: {len(train_loader)}")
        print(f"Test batches: {len(test_loader)}")
        print(f"Size of each image (flattened): {train_data[0][0].shape[0]}")

    return train_loader, test_loader