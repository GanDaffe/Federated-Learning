import torch

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, input_data, targets, transform=None):
        self.data = [input_data[i].unsqueeze(0).float() for i in range(input_data.size(0))]
        self.targets = targets
        self.classes = torch.unique(targets)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]