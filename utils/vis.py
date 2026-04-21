import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

class ResultVisualizer:
    def __init__(self):
        self.colors = sns.color_palette('tab10')
    
    def plot_dgh_distribution(self, dgh_values, save_path=None, title='ΔG_H Distribution'):
        plt.figure(figsize=(10, 6))
        sns.histplot(dgh_values, bins=30, kde=True, color=self.colors[0])
        plt.axvline(x=0, color='red', linestyle='--', label='Target (0 eV)')
        plt.xlabel('ΔG_H (eV)')
        plt.ylabel('Frequency')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_stability_vs_synthesis(self, stability_scores, synthesis_scores, save_path=None):
        plt.figure(figsize=(10, 8))
        sns.scatterplot(x=stability_scores, y=synthesis_scores, alpha=0.7, color=self.colors[1])
        plt.xlabel('Stability Score')
        plt.ylabel('Synthesis Probability')
        plt.title('Stability vs Synthesis Probability')
        plt.grid(True, alpha=0.3)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_loss_curve(self, train_losses, val_losses=None, save_path=None):
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Training Loss', color=self.colors[0])
        if val_losses:
            plt.plot(val_losses, label='Validation Loss', color=self.colors[1])
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_material_structure_2d(self, positions, atom_types, save_path=None, title='Generated Structure'):
        plt.figure(figsize=(8, 8))
        
        element_colors = {
            'H': 'white', 'C': 'gray', 'N': 'blue', 'O': 'red',
            'B': 'pink', 'P': 'orange', 'S': 'yellow',
            'Fe': 'orange', 'Co': 'gray', 'Ni': 'gray',
            'Cu': 'orange', 'Ag': 'gray', 'Au': 'yellow',
            'Pt': 'gray', 'Pd': 'gray', 'Rh': 'gray',
            'Ir': 'gray', 'Ru': 'gray'
        }
        
        for i, (pos, atom_type) in enumerate(zip(positions, atom_types)):
            color = element_colors.get(atom_type, 'gray')
            plt.scatter(pos[0], pos[1], s=200, c=color, edgecolor='black', zorder=10)
            plt.text(pos[0], pos[1] + 0.1, atom_type, fontsize=10, ha='center')
        
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_material_structure_3d(self, positions, atom_types, save_path=None, title='3D Structure'):
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        element_colors = {
            'H': 'white', 'C': 'gray', 'N': 'blue', 'O': 'red',
            'B': 'pink', 'P': 'orange', 'S': 'yellow',
            'Fe': 'orange', 'Co': 'gray', 'Ni': 'gray',
            'Cu': 'orange', 'Ag': 'gray', 'Au': 'yellow',
            'Pt': 'gray', 'Pd': 'gray', 'Rh': 'gray',
            'Ir': 'gray', 'Ru': 'gray'
        }
        
        for i, (pos, atom_type) in enumerate(zip(positions, atom_types)):
            color = element_colors.get(atom_type, 'gray')
            ax.scatter(pos[0], pos[1], pos[2], s=300, c=color, edgecolor='black')
            ax.text(pos[0], pos[1], pos[2] + 0.1, atom_type, fontsize=8)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(title)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_her_performance(self, dgh_values, baseline_dgh=None, save_path=None):
        plt.figure(figsize=(12, 6))
        
        sorted_dgh = sorted(dgh_values)
        plt.plot(range(len(sorted_dgh)), sorted_dgh, marker='o', linestyle='-', 
                 label='Generated Materials', color=self.colors[0])
        
        if baseline_dgh is not None:
            plt.axhline(y=baseline_dgh, color='red', linestyle='--', 
                       label=f'Baseline Avg: {baseline_dgh:.2f} eV')
        
        plt.axhline(y=0, color='green', linestyle=':', label='Target (0 eV)')
        
        plt.xlabel('Material Index')
        plt.ylabel('ΔG_H (eV)')
        plt.title('HER Catalytic Performance of Generated Materials')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim(-1.0, 1.0)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_stability_curve(self, stability_scores, synthesis_scores, save_path=None):
        plt.figure(figsize=(10, 6))
        
        indices = range(len(stability_scores))
        plt.plot(indices, stability_scores, marker='o', label='Thermodynamic Stability', 
                 color=self.colors[0])
        plt.plot(indices, synthesis_scores, marker='s', label='Synthesis Probability', 
                 color=self.colors[1])
        
        plt.xlabel('Material Index')
        plt.ylabel('Score')
        plt.title('Stability and Synthesis Probability')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_comparison_bar(self, metrics, labels, save_path=None):
        plt.figure(figsize=(10, 6))
        x = np.arange(len(labels))
        width = 0.35
        
        plt.bar(x - width/2, metrics['baseline'], width, label='Baseline', color=self.colors[0])
        plt.bar(x + width/2, metrics['ours'], width, label='Ours', color=self.colors[1])
        
        plt.xlabel('Metrics')
        plt.ylabel('Score')
        plt.title('Comparison with Baseline')
        plt.xticks(x, labels)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_pareto_front(self, dgh_values, stability_scores, save_path=None):
        plt.figure(figsize=(10, 8))
        sns.scatterplot(x=[abs(d) for d in dgh_values], y=stability_scores, 
                       alpha=0.7, color=self.colors[0])
        plt.xlabel('|ΔG_H| (eV)')
        plt.ylabel('Stability Score')
        plt.title('Pareto Front: HER Activity vs Stability')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

class StructureVisualizer:
    @staticmethod
    def visualize_structure(structure, save_path=None, show_3d=True):
        positions = []
        atom_types = []
        
        for site in structure:
            positions.append(site.coords)
            atom_types.append(str(site.specie))
        
        positions = np.array(positions)
        
        if show_3d:
            ResultVisualizer().plot_material_structure_3d(positions, atom_types, save_path)
        else:
            ResultVisualizer().plot_material_structure_2d(positions, atom_types, save_path)
    
    @staticmethod
    def visualize_multiple_structures(structures, save_dir='results', num_cols=3):
        num_rows = (len(structures) + num_cols - 1) // num_cols
        
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(5*num_cols, 5*num_rows))
        axes = axes.flatten()
        
        element_colors = {
            'H': 'white', 'C': 'gray', 'N': 'blue', 'O': 'red',
            'B': 'pink', 'P': 'orange', 'S': 'yellow',
            'Fe': 'orange', 'Co': 'gray', 'Ni': 'gray',
            'Cu': 'orange', 'Ag': 'gray', 'Au': 'yellow',
            'Pt': 'gray', 'Pd': 'gray', 'Rh': 'gray',
            'Ir': 'gray', 'Ru': 'gray'
        }
        
        for i, (structure, ax) in enumerate(zip(structures, axes)):
            positions = np.array([site.coords for site in structure])
            atom_types = [str(site.specie) for site in structure]
            
            for pos, atom_type in zip(positions, atom_types):
                color = element_colors.get(atom_type, 'gray')
                ax.scatter(pos[0], pos[1], s=150, c=color, edgecolor='black')
            
            ax.set_title(f'Structure {i+1}')
            ax.set_aspect('equal')
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/generated_structures.png', dpi=300)
        plt.close()