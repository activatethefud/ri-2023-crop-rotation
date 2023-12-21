import json
import numpy as np

class CRSP:
    def __init__(self, problem_filename):
        self.load_problem(problem_filename)

        self.solution = np.zeros((
            self.K,
            self.N,
            self.M
        ))
    
    def load_problem(self, problem_filename: str):

        with open(problem_filename, 'r') as f:
            self.problem_definition = json.load(f)

            self.M = self.problem_definition["time_units"]
            self.K = len(self.problem_definition["plot_adjacency"])
            self.crops = self.problem_definition["crops"]
            self.crops_idx = list(range(len(self.crops)))

            self.N = len(self.crops)

            # divide crops into family sets
            crop_families = list(set(c["family"] for c in self.crops))
            family_mapping = { crop_families[i]: i for i in range(len(crop_families))}

            self.B = [ [] for _ in range(len(crop_families))] # sets of crops based on family

            for idx in self.crops_idx:
                self.B[family_mapping[self.crops[idx]["family"]]].append(
                    idx
                )
            
            self.T = [ c["grow_time"] for c in self.crops ] # grow times for crop c
            self.F = [ family_mapping[c["family"]] for c in self.crops ] # family for crop c
            self.I = [ c["planting_month"] for c in self.crops ] # planting periods for crop c
            self.C = self.crops_idx

    def objective(self):

        score = 0

        for k in range(self.K):
            for c in self.C:
                for t in self.I[c]:
                    score += self.solution[k][c][t] * self.T[c]

        return 0

    def constraint1(self):
        pass
    
    def constraint2(self):
        pass
    
    def constraint3(self):
        pass
    
    def constraint4(self):
        pass
    
    def constraint5(self):
        pass
    
    def solve(self):
        pass