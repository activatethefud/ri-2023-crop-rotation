import json
import numpy as np

class CRSP:
    def __init__(self, problem_filename):
        self.load_problem(problem_filename)

        self.solution = np.zeros((
            len(self.crops),
            self.M,
            self.K
        ))
    
    def load_problem(self, problem_filename: str):

        with open(problem_filename, 'r') as f:
            self.problem_definition = json.load(f)

            self.M = self.problem_definition["time_units"]
            self.K = len(self.problem_definition["plot_adjacency"])
            self.crops = self.problem_definition["crops"]
        
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