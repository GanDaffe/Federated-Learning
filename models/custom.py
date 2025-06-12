import torch

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, input_data, targets, transform=None):
        self.data = input_data
        self.targets = targets
        self.classes = torch.unique(torch.tensor(targets)).tolist()


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]