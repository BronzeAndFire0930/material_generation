import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from torch_geometric.data import Data, Batch

class EdgeConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='max')
        self.mlp = nn.Sequential(
            nn.Linear(2 * in_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels)
        )

    def forward(self, x, edge_index):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        return self.propagate(edge_index, x=x)

    def message(self, x_i, x_j):
        tmp = torch.cat([x_i, x_j - x_i], dim=1)
        return self.mlp(tmp)

class DiffusionEncoder(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim=128, num_layers=4):
        super().__init__()
        self.node_embedding = nn.Linear(node_dim, hidden_dim)
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(EdgeConv(hidden_dim, hidden_dim))
        self.fc = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, data):
        x = self.node_embedding(data.x)
        for layer in self.layers:
            x = layer(x, data.edge_index)
        x = self.fc(x)
        return x

class DiffusionDecoder(nn.Module):
    def __init__(self, hidden_dim, node_dim, num_layers=4):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(EdgeConv(hidden_dim, hidden_dim))
        self.fc = nn.Linear(hidden_dim, node_dim)

    def forward(self, x, edge_index):
        for layer in self.layers:
            x = layer(x, edge_index)
        x = self.fc(x)
        return x

class ScoreNetwork(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim=128):
        super().__init__()
        self.encoder = DiffusionEncoder(node_dim, edge_dim, hidden_dim)
        self.decoder = DiffusionDecoder(hidden_dim, node_dim)
        self.node_dim = node_dim
        self.time_proj = nn.Linear(64, node_dim)

    def forward(self, data, t):
        t_emb = self.time_embedding(t)
        t_emb = self.time_proj(t_emb)
        x = data.x + t_emb.unsqueeze(0).repeat(data.x.size(0), 1)
        data_modified = Data(x=x, edge_index=data.edge_index)
        latent = self.encoder(data_modified)
        score = self.decoder(latent, data.edge_index)
        return score

    def time_embedding(self, t):
        freq = torch.exp(torch.linspace(-4, 4, 32))
        emb = torch.concat([torch.sin(t * freq), torch.cos(t * freq)], dim=-1)
        return emb

class CrystalDiffusionModel(nn.Module):
    def __init__(self, node_dim=128, edge_dim=64, hidden_dim=128, noise_schedule='cosine'):
        super().__init__()
        self.score_net = ScoreNetwork(node_dim, edge_dim, hidden_dim)
        self.noise_schedule = noise_schedule
        self.node_dim = node_dim

    def cosine_schedule(self, timesteps=1000):
        s = 0.008
        steps = torch.linspace(0, timesteps, timesteps + 1)
        alpha_bar = torch.cos((steps / timesteps + s) / (1 + s) * torch.pi / 2) ** 2
        alpha = alpha_bar[1:] / alpha_bar[:-1]
        return alpha, alpha_bar

    def sample(self, edge_index, num_nodes, device='cpu', timesteps=1000):
        alpha, alpha_bar = self.cosine_schedule(timesteps)
        x = torch.randn(num_nodes, self.node_dim, device=device)
        
        for t in reversed(range(timesteps)):
            if t > 0:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)
            
            data = Data(x=x, edge_index=edge_index.to(device))
            score = self.score_net(data, torch.tensor(t / timesteps, device=device))
            
            beta_t = 1 - alpha[t]
            x = (x - beta_t * score) / torch.sqrt(alpha[t]) + torch.sqrt(beta_t) * noise
        
        return x

    def forward(self, data):
        timesteps = 1000
        alpha, alpha_bar = self.cosine_schedule(timesteps)
        
        if hasattr(data, 'num_graphs'):
            num_graphs = data.num_graphs
        else:
            num_graphs = 1
        
        t = torch.randint(0, timesteps, (num_graphs,), device=data.x.device)
        alpha_bar_t = alpha_bar[t].unsqueeze(-1).unsqueeze(-1)
        
        if num_graphs == 1:
            alpha_bar_t = alpha_bar_t.squeeze(0)
        
        noise = torch.randn_like(data.x)
        x_noisy = torch.sqrt(alpha_bar_t) * data.x + torch.sqrt(1 - alpha_bar_t) * noise
        
        data_noisy = Data(x=x_noisy, edge_index=data.edge_index)
        
        if num_graphs == 1:
            score = self.score_net(data_noisy, t.float() / timesteps)
        else:
            score = self.score_net(data_noisy, t.float() / timesteps)
        
        loss = F.mse_loss(score, noise)
        return loss