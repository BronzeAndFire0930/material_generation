import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch
import numpy as np
from pymatgen.core import Structure, Lattice

class StructureGenerator(nn.Module):
    def __init__(self, diffusion_model, node_dim=128):
        super().__init__()
        self.diffusion_model = diffusion_model
        self.node_dim = node_dim
        self.position_decoder = nn.Linear(node_dim, 3)
        self.atom_type_decoder = nn.Linear(node_dim, 100)

    def generate_structure(self, edge_index, num_nodes, device='cpu'):
        latent = self.diffusion_model.sample(edge_index, num_nodes, device)
        positions = self.position_decoder(latent)
        atom_logits = self.atom_type_decoder(latent)
        atom_types = torch.argmax(atom_logits, dim=-1)
        return positions, atom_types, latent

    def forward(self, data):
        batch_size = data.num_graphs
        all_positions = []
        all_atom_types = []
        
        for i in range(batch_size):
            mask = data.batch == i
            edge_index_i = data.edge_index[:, mask[data.edge_index[0]] & mask[data.edge_index[1]]]
            num_nodes_i = mask.sum().item()
            
            positions, atom_types, _ = self.generate_structure(edge_index_i, num_nodes_i, data.x.device)
            all_positions.append(positions)
            all_atom_types.append(atom_types)
        
        return all_positions, all_atom_types

class CrystalStructureGenerator:
    def __init__(self, model):
        self.model = model
        
    def generate_crystal(self, edge_index, num_nodes, device='cpu'):
        self.model.eval()
        with torch.no_grad():
            positions, atom_types, latent = self.model.generate_structure(
                edge_index, num_nodes, device
            )
        
        positions = positions.cpu().numpy()
        atom_types = atom_types.cpu().numpy()
        
        atom_symbols = self._convert_atom_types(atom_types)
        lattice = Lattice.cubic(3.5)
        
        structure = Structure(
            lattice=lattice,
            species=atom_symbols,
            coords=positions,
            coords_are_cartesian=True
        )
        
        return structure, latent

    def _convert_atom_types(self, atom_types):
        periodic_table = [
            'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
            'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca',
            'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
            'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr',
            'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn',
            'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd',
            'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb',
            'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
            'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th',
            'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm'
        ]
        return [periodic_table[int(t)] for t in atom_types % len(periodic_table)]

    def generate_multiple_structures(self, num_structures=10, num_nodes_range=(6, 12), device='cpu'):
        structures = []
        for _ in range(num_structures):
            num_nodes = np.random.randint(*num_nodes_range)
            edge_index = self._generate_edge_index(num_nodes)
            structure, _ = self.generate_crystal(edge_index, num_nodes, device)
            structures.append(structure)
        return structures

    def _generate_edge_index(self, num_nodes):
        edges = []
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                if np.random.random() < 0.5:
                    edges.append([i, j])
                    edges.append([j, i])
        if not edges:
            edges = [[i, (i + 1) % num_nodes] for i in range(num_nodes)]
            edges += [[(i + 1) % num_nodes, i] for i in range(num_nodes)]
        return torch.tensor(edges, dtype=torch.long).t()