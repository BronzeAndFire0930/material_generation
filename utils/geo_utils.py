import torch
import numpy as np
from pymatgen.core import Structure, Lattice
from pymatgen.analysis.adsorption import AdsorbateSiteFinder
from pymatgen.electronic_structure.bandstructure import BandStructure
from pymatgen.core.surface import SlabGenerator, generate_all_slabs

class HERCalculator:
    @staticmethod
    def calculate_dgh(structure, method='approximate'):
        if method == 'approximate':
            return HERCalculator._approximate_dgh(structure)
        else:
            return HERCalculator._detailed_dgh(structure)

    @staticmethod
    def _approximate_dgh(structure):
        atom_counts = {}
        for site in structure:
            element = str(site.specie)
            atom_counts[element] = atom_counts.get(element, 0) + 1
        
        dgh = 0.0
        element_contributions = {
            'Pt': -0.09, 'Ir': -0.18, 'Rh': -0.15, 'Ru': 0.18,
            'Pd': 0.24, 'Au': 0.40, 'Ag': 0.65, 'Cu': 0.08,
            'Ni': 0.12, 'Co': 0.20, 'Fe': 0.30, 'C': 0.25,
            'N': 0.15, 'B': 0.45, 'S': 0.50, 'O': 0.80
        }
        
        total_weight = 0
        for element, count in atom_counts.items():
            if element in element_contributions:
                dgh += element_contributions[element] * count
                total_weight += count
        
        if total_weight > 0:
            dgh /= total_weight
        
        dgh += np.random.normal(0, 0.1)
        return min(max(dgh, -1.5), 1.5)

    @staticmethod
    def _detailed_dgh(structure):
        return HERCalculator._approximate_dgh(structure)

class StabilityCalculator:
    @staticmethod
    def calculate_stability(structure):
        thermo_stability = StabilityCalculator._thermodynamic_stability(structure)
        dyn_stability = StabilityCalculator._dynamic_stability(structure)
        return (thermo_stability + dyn_stability) / 2

    @staticmethod
    def _thermodynamic_stability(structure):
        lattice = structure.lattice
        volume_per_atom = lattice.volume / len(structure)
        
        ideal_volumes = {
            'Pt': 16.0, 'Ir': 15.0, 'Rh': 13.5, 'Ru': 14.0,
            'Pd': 16.5, 'Au': 16.9, 'Ag': 17.1, 'Cu': 11.8,
            'Ni': 10.9, 'Co': 10.0, 'Fe': 11.3, 'C': 8.0,
            'N': 6.0, 'B': 7.0, 'S': 15.0, 'O': 8.0
        }
        
        avg_ideal_vol = 0.0
        count = 0
        for site in structure:
            element = str(site.specie)
            if element in ideal_volumes:
                avg_ideal_vol += ideal_volumes[element]
                count += 1
        
        if count > 0:
            avg_ideal_vol /= count
            vol_ratio = volume_per_atom / avg_ideal_vol
            thermo_score = max(0.0, 1.0 - abs(vol_ratio - 1.0) * 2)
        else:
            thermo_score = 0.7 + np.random.normal(0, 0.1)
        
        return min(max(thermo_score, 0.0), 1.0)

    @staticmethod
    def _dynamic_stability(structure):
        bond_angles = []
        for i, site_i in enumerate(structure):
            for j, site_j in enumerate(structure):
                if i < j:
                    dist = site_i.distance(site_j)
                    if dist < 3.0:
                        for k, site_k in enumerate(structure):
                            if k != i and k != j:
                                vec_ij = site_j.coords - site_i.coords
                                vec_ik = site_k.coords - site_i.coords
                                angle = np.arccos(np.dot(vec_ij, vec_ik) / 
                                                  (np.linalg.norm(vec_ij) * np.linalg.norm(vec_ik)))
                                bond_angles.append(np.degrees(angle))
        
        if bond_angles:
            avg_angle = np.mean(bond_angles)
            dyn_score = max(0.0, 1.0 - abs(avg_angle - 120.0) / 60.0)
        else:
            dyn_score = 0.6 + np.random.normal(0, 0.1)
        
        return min(max(dyn_score, 0.0), 1.0)

class SynthesisPredictor:
    @staticmethod
    def predict_synthesis_probability(structure):
        score = 0.0
        
        element_counts = {}
        for site in structure:
            element = str(site.specie)
            element_counts[element] = element_counts.get(element, 0) + 1
        
        common_elements = ['C', 'N', 'O', 'S', 'B', 'P', 'Si', 
                          'Cu', 'Ni', 'Co', 'Fe', 'Pt', 'Pd', 'Au', 'Ag']
        rare_elements = ['Ir', 'Rh', 'Ru', 'Os', 'Re', 'Ta', 'Hf']
        
        for element, count in element_counts.items():
            if element in common_elements:
                score += 0.1 * count
            elif element in rare_elements:
                score -= 0.15 * count
        
        num_atoms = len(structure)
        if 4 <= num_atoms <= 20:
            score += 0.3
        elif num_atoms < 4 or num_atoms > 30:
            score -= 0.2
        
        lattice = structure.lattice
        if lattice is not None:
            if lattice.volume < 50 or lattice.volume > 500:
                score -= 0.1
        
        score = min(max(score, 0.0), 1.0)
        score += np.random.normal(0, 0.05)
        
        return min(max(score, 0.0), 1.0)

class MaterialEvaluator:
    def __init__(self):
        self.her_calculator = HERCalculator()
        self.stability_calculator = StabilityCalculator()
        self.synthesis_predictor = SynthesisPredictor()

    def evaluate(self, structure):
        dgh = self.her_calculator.calculate_dgh(structure)
        stability = self.stability_calculator.calculate_stability(structure)
        synthesis = self.synthesis_predictor.predict_synthesis_probability(structure)
        
        return {
            'dgh': dgh,
            'stability': stability,
            'synthesis': synthesis,
            'overall_score': (1 - abs(dgh)) * 0.4 + stability * 0.3 + synthesis * 0.3
        }

    def evaluate_batch(self, structures):
        results = []
        for structure in structures:
            result = self.evaluate(structure)
            results.append(result)
        return results

class StructureAnalyzer:
    @staticmethod
    def analyze_structure(structure):
        analysis = {
            'num_atoms': len(structure),
            'elements': list(set(str(site.specie) for site in structure)),
            'lattice_parameters': structure.lattice.parameters if structure.lattice else None,
            'volume': structure.volume if structure.lattice else None,
            'density': structure.density if structure.lattice else None
        }
        return analysis