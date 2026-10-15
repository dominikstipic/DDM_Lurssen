import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset
import matplotlib.pyplot as plt

class BlueprintDataset(Dataset):
    path = "data/blueprints"
    transform = transforms.Compose([
        transforms.ToTensor(),          # Converts PIL image to [0, 1] tensor of shape (C, H, W)
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Maps range to [-1, 1]
    ])

    def __init__(self, height=640, width=640):
        self.height = height
        self.width = width
        data_dir = Path(self.path)
        self.images = list(data_dir.rglob("*"))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert("RGB")
        tensor_image = self.transform(image).unsqueeze(0)
        return tensor_image


class CustomCIFARDataset(Dataset):
    def __init__(self, root="./data", train=True, transform=None):

        # Download and load CIFAR-10
        self.dataset = torchvision.datasets.CIFAR10(
            root=root,
            train=train,
            download=True
        )

        self.transform = transform

        self.classes = self.dataset.classes

    def __len__(self):
        # Return the number of images
        return len(self.dataset)

    def sample(self, percentage, seed=None):
        if not 0 < percentage <= 100:
            raise ValueError("Percentage must be in (0, 100].")

        n_samples = int(len(self) * percentage / 100)

        generator = torch.Generator()

        if seed is not None:
            generator.manual_seed(seed)

        indices = torch.randperm(
            len(self),
            generator=generator
        )[:n_samples].tolist()

        return Subset(self, indices)

    def sample_num(self, num, seed=None):
            n_samples = num
    
            generator = torch.Generator()
    
            if seed is not None:
                generator.manual_seed(seed)
    
            indices = torch.randperm(
                len(self),
                generator=generator
            )[:n_samples].tolist()
    
            return Subset(self, indices)

    def __getitem__(self, idx):

        # Get image and label
        image, label = self.dataset[idx]

        # Apply transformations if provided
        if self.transform is not None:
            image = self.transform(image)

        return image, label

if __name__ == "__main__":
    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5]
            )
    ])
    dataset = CustomCIFARDataset(transform=transform)
    img,_ = dataset[0]
    plt.imshow(img.squeeze().permute(2,1,0))
    plt.show()

    db_sample = dataset.sample_num(1)
    print(len(db_sample))
