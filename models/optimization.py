import torch
import torch.nn as nn
import numpy as np

class HERLoss(nn.Module):
    def __init__(self, target_dg=0.0):
        super().__init__()
        self.target_dg = target_dg

    def forward(self, dg_h_values):
        loss = torch.mean((dg_h_values - self.target_dg) ** 2)
        return loss

class StabilityLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, stability_scores):
        loss = torch.mean(1.0 - stability_scores)
        return loss

class SynthesisLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, synthesis_scores):
        loss = torch.mean(1.0 - synthesis_scores)
        return loss

class MultiTaskLoss(nn.Module):
    def __init__(self, weights=[1.0, 1.0, 1.0]):
        super().__init__()
        self.her_loss = HERLoss()
        self.stability_loss = StabilityLoss()
        self.synthesis_loss = SynthesisLoss()
        self.weights = weights

    def forward(self, dg_h_values, stability_scores, synthesis_scores):
        her = self.her_loss(dg_h_values)
        stability = self.stability_loss(stability_scores)
        synthesis = self.synthesis_loss(synthesis_scores)
        total_loss = (self.weights[0] * her + 
                     self.weights[1] * stability + 
                     self.weights[2] * synthesis)
        return total_loss, {'her': her.item(), 'stability': stability.item(), 'synthesis': synthesis.item()}

class HEROptimizer:
    def __init__(self, model, lr=1e-4):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    def optimize(self, data, dg_h_target=0.0, num_iterations=100):
        self.model.train()
        losses = []
        
        for i in range(num_iterations):
            self.optimizer.zero_grad()
            
            loss = self.model(data)
            
            loss.backward()
            self.optimizer.step()
            
            losses.append(loss.item())
            
            if i % 10 == 0:
                print(f"Iteration {i}, Loss: {loss.item():.4f}")
        
        return losses

class StructureOptimizer:
    def __init__(self, model, structure_generator):
        self.model = model
        self.structure_generator = structure_generator
        self.optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    def optimize_structure(self, target_dg=0.0, num_steps=50):
        self.model.train()
        
        for step in range(num_steps):
            self.optimizer.zero_grad()
            
            edge_index = torch.tensor([[0, 1, 1, 2, 2, 0], [1, 0, 2, 1, 0, 2]], dtype=torch.long)
            num_nodes = 3
            
            positions, atom_types, latent = self.structure_generator.generate_structure(
                edge_index, num_nodes, 'cpu'
            )
            
            dg_h = self.predict_dgh(latent)
            loss = (dg_h - target_dg) ** 2
            
            loss.backward()
            self.optimizer.step()
            
            if step % 10 == 0:
                print(f"Step {step}, ΔG_H: {dg_h.item():.4f}, Loss: {loss.item():.4f}")

    def predict_dgh(self, latent):
        return torch.sum(latent) * 0.1 + torch.randn(1) * 0.05

class GeneticOptimizer:
    def __init__(self, population_size=50, mutation_rate=0.1):
        self.population_size = population_size
        self.mutation_rate = mutation_rate

    def initialize_population(self, num_nodes=6):
        population = []
        for _ in range(self.population_size):
            structure = {
                'positions': np.random.rand(num_nodes, 3),
                'atom_types': np.random.randint(1, 50, num_nodes),
                'lattice': np.eye(3) * 3.5
            }
            population.append(structure)
        return population

    def fitness(self, structure, predictor):
        dg_h = predictor.predict_dgh(structure)
        stability = predictor.predict_stability(structure)
        synthesis = predictor.predict_synthesis(structure)
        
        score = -abs(dg_h) + stability + synthesis
        return score

    def select(self, population, fitness_scores):
        sorted_indices = np.argsort(fitness_scores)[::-1]
        return [population[i] for i in sorted_indices[:self.population_size//2]]

    def crossover(self, parent1, parent2):
        child = {
            'positions': (parent1['positions'] + parent2['positions']) / 2,
            'atom_types': np.where(np.random.rand(len(parent1['atom_types'])) > 0.5, 
                                  parent1['atom_types'], parent2['atom_types']),
            'lattice': (parent1['lattice'] + parent2['lattice']) / 2
        }
        return child

    def mutate(self, structure):
        if np.random.random() < self.mutation_rate:
            structure['positions'] += np.random.randn(*structure['positions'].shape) * 0.01
        if np.random.random() < self.mutation_rate:
            idx = np.random.randint(len(structure['atom_types']))
            structure['atom_types'][idx] = np.random.randint(1, 50)
        return structure

    def optimize(self, predictor, generations=100):
        population = self.initialize_population()
        
        for gen in range(generations):
            fitness_scores = [self.fitness(s, predictor) for s in population]
            parents = self.select(population, fitness_scores)
            
            new_population = []
            while len(new_population) < self.population_size:
                parent1, parent2 = np.random.choice(parents, 2, replace=False)
                child = self.crossover(parent1, parent2)
                child = self.mutate(child)
                new_population.append(child)
            
            population = new_population
            
            if gen % 10 == 0:
                best_score = max(fitness_scores)
                print(f"Generation {gen}, Best Fitness: {best_score:.4f}")
        
        fitness_scores = [self.fitness(s, predictor) for s in population]
        best_idx = np.argmax(fitness_scores)
        return population[best_idx]