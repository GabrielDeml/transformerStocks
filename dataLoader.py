import torch
from torch.utils.data import Dataset
import glob
import pandas as pd

class CustomDataset(Dataset):

    def __init__(self, root):
        self.root = root

        # Make a list containing the path to all your csv files
        self.paths = glob.glob(f'{self.root}/**/*.csv', recursive=True)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):

        data = pd.read_csv(self.paths[idx])

        # Using 'Low', 'Open', 'Volume', 'High', 'Close' as features and 'Adjusted Close' as label
        x = torch.tensor(data[['Low', 'Open', 'Volume', 'High', 'Close']].values, dtype=torch.float32)
        y = torch.tensor(data['Adjusted Close'].values, dtype=torch.float32)

        return x, y

# Example usage
root_path = 'stock_market_data/nasdaq/csv/'
dataset = CustomDataset(root_path)

for x, y in dataset:
    print(x, y)
