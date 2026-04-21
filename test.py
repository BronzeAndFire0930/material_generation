import torch
import argparse
import os
import numpy as np
from models.diffusion_model import CrystalDiffusionModel
from models.structure_generator import StructureGenerator, CrystalStructureGenerator
from utils.geo_utils import MaterialEvaluator
from utils.vis import ResultVisualizer, StructureVisualizer

def parse_args():
    parser = argparse.ArgumentParser(description='Test diffusion model for 2D material generation')
    parser.add_argument('--model_path', type=str, default='models/pretrained/model.pt', help='Path to trained model')
    parser.add_argument('--num_structures', type=int, default=10, help='Number of structures to generate')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use')
    parser.add_argument('--save_dir', type=str, default='results', help='Directory to save results')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Hidden dimension')
    parser.add_argument('--node_dim', type=int, default=128, help='Node feature dimension')
    return parser.parse_args()

def test():
    args = parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    device = torch.device(args.device)
    
    print(f"Loading model from {args.model_path}...")
    model = CrystalDiffusionModel(
        node_dim=args.node_dim,
        hidden_dim=args.hidden_dim
    ).to(device)
    
    try:
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        print("Model loaded successfully")
    except FileNotFoundError:
        print(f"Model file not found at {args.model_path}. Using randomly initialized model.")
    
    model.eval()
    
    print(f"Generating {args.num_structures} structures...")
    structure_generator = CrystalStructureGenerator(StructureGenerator(model))
    structures = structure_generator.generate_multiple_structures(
        num_structures=args.num_structures,
        device=device
    )
    
    print("Evaluating generated structures...")
    evaluator = MaterialEvaluator()
    results = evaluator.evaluate_batch(structures)
    
    dgh_values = [r['dgh'] for r in results]
    stability_scores = [r['stability'] for r in results]
    synthesis_scores = [r['synthesis'] for r in results]
    overall_scores = [r['overall_score'] for r in results]
    
    print("\n=== Evaluation Results ===")
    print(f"Number of structures: {len(structures)}")
    print(f"Average ΔG_H: {np.mean(dgh_values):.4f} eV (Target: 0 eV)")
    print(f"ΔG_H std: {np.std(dgh_values):.4f}")
    print(f"Best ΔG_H: {min(dgh_values, key=abs):.4f} eV")
    print(f"\nAverage Stability: {np.mean(stability_scores):.4f}")
    print(f"Average Synthesis Probability: {np.mean(synthesis_scores):.4f}")
    print(f"Average Overall Score: {np.mean(overall_scores):.4f}")
    
    visualizer = ResultVisualizer()
    
    visualizer.plot_dgh_distribution(dgh_values,
                                     save_path=os.path.join(args.save_dir, 'dgh_distribution.png'))
    
    visualizer.plot_her_performance(dgh_values,
                                    save_path=os.path.join(args.save_dir, 'her_performance.png'))
    
    visualizer.plot_stability_vs_synthesis(stability_scores, synthesis_scores,
                                           save_path=os.path.join(args.save_dir, 'stability_vs_synthesis.png'))
    
    visualizer.plot_stability_curve(stability_scores, synthesis_scores,
                                    save_path=os.path.join(args.save_dir, 'stability_curve.png'))
    
    visualizer.plot_pareto_front(dgh_values, stability_scores,
                                 save_path=os.path.join(args.save_dir, 'pareto_front.png'))
    
    StructureVisualizer.visualize_multiple_structures(structures, save_dir=args.save_dir)
    
    print("\n=== Saving Results ===")
    results_data = {
        'dgh_values': dgh_values,
        'stability_scores': stability_scores,
        'synthesis_scores': synthesis_scores,
        'overall_scores': overall_scores
    }
    np.save(os.path.join(args.save_dir, 'evaluation_results.npy'), results_data)
    print(f"Results saved to {args.save_dir}")
    
    print("\n=== Best Performing Structures ===")
    sorted_indices = np.argsort([abs(d) for d in dgh_values])
    for i in range(min(5, len(structures))):
        idx = sorted_indices[i]
        print(f"\nStructure {i+1}:")
        print(f"  ΔG_H: {dgh_values[idx]:.4f} eV")
        print(f"  Stability: {stability_scores[idx]:.4f}")
        print(f"  Synthesis Probability: {synthesis_scores[idx]:.4f}")
        print(f"  Overall Score: {overall_scores[idx]:.4f}")

if __name__ == '__main__':
    test()