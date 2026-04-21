import os
import torch
import numpy as np
from torch_geometric.data import Dataset, Data
from pymatgen.core import Structure
from pymatgen.io.cif import CifParser
from sklearn.model_selection import train_test_split

class CrystalDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.data_list = []
        self._load_data()

    @property
    def raw_file_names(self):
        return [f for f in os.listdir(self.raw_dir) if f.endswith('.cif')]

    @property
    def processed_file_names(self):
        return ['data.pt']

    def _load_data(self):
        cif_files = self.raw_file_names
        
        if not cif_files:
            print("No CIF files found. Generating synthetic data...")
            self._generate_synthetic_data()
            return

        for cif_file in cif_files:
            try:
                parser = CifParser(os.path.join(self.raw_dir, cif_file))
                structure = parser.get_structures()[0]
                data = self._structure_to_data(structure)
                self.data_list.append(data)
            except Exception as e:
                print(f"Failed to load {cif_file}: {e}")

        if not self.data_list:
            print("No valid CIF files found. Generating synthetic data...")
            self._generate_synthetic_data()

    def _generate_synthetic_data(self):
        np.random.seed(42)
        num_structures = 100
        
        for i in range(num_structures):
            num_nodes = np.random.randint(4, 12)
            edge_index = self._generate_edge_index(num_nodes)
            
            x = torch.randn(num_nodes, 128)
            pos = torch.randn(num_nodes, 3)
            
            dg_h = torch.tensor([np.random.uniform(-1.0, 1.0)])
            stability = torch.tensor([np.random.uniform(0.5, 1.0)])
            synthesis = torch.tensor([np.random.uniform(0.3, 0.9)])
            
            data = Data(
                x=x,
                edge_index=edge_index,
                pos=pos,
                dg_h=dg_h,
                stability=stability,
                synthesis=synthesis
            )
            self.data_list.append(data)

    def _generate_edge_index(self, num_nodes):
        edges = []
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j and np.random.random() < 0.4:
                    edges.append([i, j])
        if not edges:
            edges = [[i, (i + 1) % num_nodes] for i in range(num_nodes)]
        return torch.tensor(edges, dtype=torch.long).t()

    def _structure_to_data(self, structure):
        num_nodes = len(structure)
        
        atom_features = []
        for site in structure:
            atomic_num = site.specie.Z
            features = np.zeros(100)
            features[atomic_num - 1] = 1
            atom_features.append(features)
        
        x = torch.tensor(atom_features, dtype=torch.float)
        pos = torch.tensor(structure.cart_coords, dtype=torch.float)
        
        edge_index = self._generate_edge_index(num_nodes)
        
        dg_h = torch.tensor([0.0])
        stability = torch.tensor([0.8])
        synthesis = torch.tensor([0.7])
        
        data = Data(
            x=x,
            edge_index=edge_index,
            pos=pos,
            dg_h=dg_h,
            stability=stability,
            synthesis=synthesis
        )
        return data

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        return self.data_list[idx]

    def split_dataset(self, test_size=0.2, val_size=0.1):
        indices = np.arange(len(self))
        train_indices, test_indices = train_test_split(indices, test_size=test_size)
        train_indices, val_indices = train_test_split(train_indices, test_size=val_size)
        
        train_dataset = [self[i] for i in train_indices]
        val_dataset = [self[i] for i in val_indices]
        test_dataset = [self[i] for i in test_indices]
        
        return train_dataset, val_dataset, test_dataset

class MaterialDataLoader:
    def __init__(self, dataset, batch_size=32, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        indices = np.arange(len(self.dataset))
        if self.shuffle:
            np.random.shuffle(indices)
        
        for i in range(0, len(self.dataset), self.batch_size):
            batch_indices = indices[i:i+self.batch_size]
            batch = [self.dataset[j] for j in batch_indices]
            yield batch

    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size

def create_dataloaders(root_dir='data', batch_size=32):
    dataset = CrystalDataset(root_dir)
    train_data, val_data, test_data = dataset.split_dataset()
    
    train_loader = MaterialDataLoader(train_data, batch_size, shuffle=True)
    val_loader = MaterialDataLoader(val_data, batch_size, shuffle=False)
    test_loader = MaterialDataLoader(test_data, batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader