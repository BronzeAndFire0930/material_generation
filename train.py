import torch
import argparse
import os
import numpy as np
from tqdm import tqdm
from models.diffusion_model import CrystalDiffusionModel
from models.structure_generator import StructureGenerator, CrystalStructureGenerator
from models.optimization import MultiTaskLoss, HEROptimizer
from dataset.material_dataset import CrystalDataset, create_dataloaders
from utils.geo_utils import MaterialEvaluator
from utils.vis import ResultVisualizer

def parse_args():
    parser = argparse.ArgumentParser(description='Train diffusion model for 2D material generation')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Hidden dimension')
    parser.add_argument('--node_dim', type=int, default=128, help='Node feature dimension')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use')
    parser.add_argument('--save_dir', type=str, default='results', help='Directory to save results')
    parser.add_argument('--model_path', type=str, default='models/pretrained/model.pt', help='Path to save model')
    return parser.parse_args()

def train_epoch(model, dataloader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch in tqdm(dataloader, desc='Training'):
        optimizer.zero_grad()
        
        batch_loss = 0.0
        for data in batch:
            data = data.to(device)
            loss = model(data)
            batch_loss += loss
        
        batch_loss = batch_loss / len(batch)
        batch_loss.backward()
        optimizer.step()
        
        total_loss += batch_loss.item()
        num_batches += 1
    
    return total_loss / num_batches

def validate(model, dataloader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Validating'):
            batch_loss = 0.0
            for data in batch:
                data = data.to(device)
                loss = model(data)
                batch_loss += loss
            
            batch_loss = batch_loss / len(batch)
            total_loss += batch_loss.item()
            num_batches += 1
    
    return total_loss / num_batches

def train():
    args = parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
    
    device = torch.device(args.device)
    
    print(f"Loading dataset...")
    train_loader, val_loader, test_loader = create_dataloaders(batch_size=args.batch_size)
    
    print(f"Initializing model...")
    model = CrystalDiffusionModel(
        node_dim=args.node_dim,
        hidden_dim=args.hidden_dim
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = MultiTaskLoss()
    
    print(f"Starting training...")
    train_losses = []
    val_losses = []
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss = validate(model, val_loader, loss_fn, device)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), args.model_path)
            print(f"Model saved to {args.model_path}")
    
    torch.save(model.state_dict(), args.model_path)
    print(f"Final model saved to {args.model_path}")
    
    visualizer = ResultVisualizer()
    visualizer.plot_loss_curve(train_losses, val_losses, 
                               save_path=os.path.join(args.save_dir, 'loss_curve.png'))
    
    print("Generating sample materials...")
    structure_generator = CrystalStructureGenerator(StructureGenerator(model))
    structures = structure_generator.generate_multiple_structures(num_structures=10)
    
    evaluator = MaterialEvaluator()
    results = evaluator.evaluate_batch(structures)
    
    dgh_values = [r['dgh'] for r in results]
    stability_scores = [r['stability'] for r in results]
    synthesis_scores = [r['synthesis'] for r in results]
    
    visualizer.plot_dgh_distribution(dgh_values, 
                                     save_path=os.path.join(args.save_dir, 'dgh_distribution.png'))
    visualizer.plot_her_performance(dgh_values, 
                                    save_path=os.path.join(args.save_dir, 'her_performance.png'))
    visualizer.plot_stability_curve(stability_scores, synthesis_scores,
                                    save_path=os.path.join(args.save_dir, 'stability_curve.png'))
    
    from utils.vis import StructureVisualizer
    StructureVisualizer.visualize_multiple_structures(structures, save_dir=args.save_dir)
    
    print(f"Average ΔG_H: {np.mean(dgh_values):.4f} eV")
    print(f"Average Stability: {np.mean(stability_scores):.4f}")
    print(f"Average Synthesis Probability: {np.mean(synthesis_scores):.4f}")

if __name__ == '__main__':
    train()