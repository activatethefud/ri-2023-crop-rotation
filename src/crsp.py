import json
import numpy as np
import random as rd
import functools
import operator
import math
import matplotlib.pyplot as plt

class CRSPSolution:

    def __init__(self, N, M, K, T, I, B, mutation_strength = 0.2):

        self.N = N # Number of crops
        self.M = M # Number of periods
        self.K = K # Number of fields
        self.T = T # Vegetation period of crops (in periods)
        self.I = I # Crops planting periods
        self.B = B # Crop families seperated
        self.learning_rate = 0.2
        self.mutation_strength = mutation_strength

        self.plan = np.zeros((
            self.N,
            self.M,
            self.K
        ))
    
    def generate_plot(self):

        plot = np.zeros((self.N, self.M))
        B_order = list(range(len(self.B)))
        rd.shuffle(B_order)
        t = 0

        while t < self.M:

            time_left = self.M - t - 1

            for b in B_order:

                planted = False

                for c in self.B[b]:
                    if t in self.I[c] and time_left >= self.T[c] and rd.randint(0,1) > 0:
                        plot[c][t] = 1
                        t += self.T[c]
                        time_left -= t
                        planted = True
                        break
                
                if planted: # If a plant family is planted, then put a fallow period
                    t += self.T[-1]
                    time_left -= self.T[-1]

            
            t += 1
        
        return plot


    def initialize(self):
        for k in range(self.K):
            self.plan[:,:,k] = self.generate_plot()
        
        return self
    
    def set_plan(self, new_plan):
        self.plan = np.copy(new_plan)

    def recombination(self, parents):
        mu = len(parents)

        self.plan = functools.reduce(operator.add, (p.plan for p in parents))//mu
        self.mutation_strength = functools.reduce(operator.add, (p.mutation_strength for p in parents))/mu

        return self

    def crossover(self, parent):

        new_mutation_strength = (self.mutation_strength+parent.mutation_strength)/2

        new_sol = CRSPSolution(
            self.N,
            self.M,
            self.K,
            self.T,
            self.I,
            self.B,
            new_mutation_strength
        )

        for k in range(self.K):
            plot = self.plan[:,:,k] if rd.randint(0,1) == 0 else parent.plan[:,:,k]
            new_sol.plan[:,:,k] = plot

        #mask = np.random.randint(2, size = self.plan.shape)
        #new_sol.plan = mask*(self.plan - parent.plan) + parent.plan

        #mu = len(parents)

        return new_sol


    def mutation(self):

        for k in range(self.K):
            if(np.random.random() < self.mutation_strength):
                self.plan[:,:,k] = self.generate_plot()

        #for i in range(self.N):
        #    for j in self.I[i]:
        #        for k in range(self.K):
        #            if(np.random.random() < self.mutation_strength):
        #                self.plan[i][j][k] = not self.plan[i][j][k]

        self.mutation_strength *= np.exp(self.learning_rate * np.random.randn())
        return self

class CRSP:
    def __init__(self, problem_filename):
        self.load_problem(problem_filename)

        self.hard_penalty = -100
        self.soft_penalty = -0.1

        self.mutation_strength = 0.2
        self.population = []
        self.numof_parents = 5
        self.population_size = 20
        self.elitism = 0
        self.best_fitness = float("-inf")
        self.best_sol = None
        self.running_counter = 0
        self.running_sum = 0
        self.report_freq = 30

    def load_problem(self, problem_filename: str):

        with open(problem_filename, 'r') as f:
            self.problem_definition = json.load(f)

            self.M = self.problem_definition["time_units"]
            self.K = len(self.problem_definition["plot_adjacency"])
            self.Ka = self.problem_definition["plot_areas"]
            self.plot_adjacency = self.problem_definition["plot_adjacency"]

            self.crops = self.problem_definition["crops"]

            if self.problem_definition.get("start_month") != None:
                self.start_month = self.problem_definition["start_month"]

                crops_filtered = []

                for c in self.crops:
                    if self.start_month > max(c["planting_month"]):
                        continue
                    
                    months_filtered = [m for m in c["planting_month"] if m >= self.start_month]
                    c["planting_month"] = months_filtered
                    crops_filtered.append(c)

                self.crops = crops_filtered



            self.crops_idx = list(range(len(self.crops)))

            self.N = len(self.crops)

            # divide crops into family sets
            crop_families = list(set(c["family"] for c in self.crops if c["family"] != "fallow"))
            family_mapping = { crop_families[i]: i for i in range(len(crop_families))}

            self.B = [ [] for _ in range(len(crop_families))] # sets of crops based on family

            for idx in self.crops_idx[:-1]:
                self.B[family_mapping[self.crops[idx]["family"]]].append(
                    idx
                )
            
            self.days_per_period = 365//self.M
            self.periods_per_month = 30//self.days_per_period
            
            self.T = [ math.ceil(c["grow_time"]/self.days_per_period) for c in self.crops ] # grow times for crop c
            self.P = [ c["plants_per_hectare"] for c in self.crops ] # grow times for crop c
            self.F = [ family_mapping[c["family"]] for c in self.crops if c["family"] != "fallow" ] # family for crop c

            self.I = [[] for c in self.crops]

            for c in range(self.N):
                for pm in self.crops[c]["planting_month"]:
                    pm = (pm-1)*self.M//12
                    for period in range(pm, pm + self.periods_per_month):
                        self.I[c].append(int(period))

            #self.I = [ list(map(lambda x : (x-1)*self.M//12,c["planting_month"])) for c in self.crops ] # planting periods for crop c
            self.Y = [ c["yield"] for c in self.crops ]
            self.C = self.crops_idx
            self.D = [] # Green manure crops

            for idx, crop in enumerate(self.crops):
                if crop["name"] in ["grasak"]:
                    self.D.append(idx)


    def objective(self, sol):

        score = 0

        for k in range(self.K):
            for c in range(self.N-1):
                for t in self.I[c]:
                    #score += sol.plan[c][t][k] * self.T[c] # Time on land spent growing
                    score += sol.plan[c][t][k] * (self.Y[c]*1000/self.P[c]) * math.floor(self.P[c]*0.0001*self.Ka[k]) # Maximize yield
                    #score += sol.plan[c][t][k] * self.crops[c]["plants_per_hectare"]/100/100

        return score

    def constraint2(self, sol):

        penalty = 0

        # two crops cannot be at same place at same time
        for k in range(self.K):
            for j in range(self.M):
                _sum = 0

                for i in range(self.N-1):
                    for q in range(self.T[i]):
                        _sum += sol.plan[i][j-q][k]
                
                penalty += (_sum-1) * self.hard_penalty if _sum > 1 else 0
                
        return penalty
    
    def constraint3(self, sol):
        # Green manure constraint
        penalty = 0

        for k in range(self.K):
            _sum = 0

            for d in self.D: # manure crop idxs
                for j in self.I[d]:
                    _sum += sol.plan[d][j][k]
            
            #penalty += self.hard_penalty if _sum == 0 else 0
            penalty += self.soft_penalty if _sum == 0 else 0

        return penalty
    
    def constraint4(self, sol):
        # Must be at least one fallow period in each plot

        return 0

        penalty = 0
        for k in range(self.K):
            _sum = sol.plan[self.N-1,:,k].sum()
            #penalty += self.hard_penalty if _sum == 0 else 0
            penalty += self.soft_penalty if _sum == 0 else 0
        
        return penalty
    
    def constraint5(self,sol):
        # Two of the same family cannot be in succession (with no fallow in between)
        # This is the crop rotation constraint

        penalty = 0

        for k in range(self.K):
            for f in self.B: # Remove fallow
                for j in range(self.M):
                    _sum = 0

                    for i in f:
                        #q = self.T[i]+self.T[-1]-1
                        q = self.T[i]+self.T[-1]
                        _sum += sol.plan[i,max(0,j-q):j,k].sum()
                    
                    penalty += self.hard_penalty * _sum if _sum > 1 else 0


        return penalty
    
    def constraint6(self,sol):
        # Same botanical families should not be planted in adjacent fields

        penalty = 0

        for k in range(self.K):
            for f in self.B:
                for c1 in f:
                    q = self.T[c1]
                    for c1i in self.I[c1]:
                        _sum = 0
                        for c2 in f:
                            for k_adj in self.plot_adjacency[k]:
                                _sum += sol.plan[c1,c1i:c1i+q,k].sum() + sol.plan[c2,c1i:c1i+q,k_adj].sum()

                        penalty += self.soft_penalty * _sum if _sum > 1 else 0

        return penalty

    def constraint7(self, sol):
        # Each crop must be planted within it's planting periods
        # Variable constraints
        penalty = 0

        for i in range(self.N-1):
            Ic = np.array([ic for ic in range(self.M) if ic not in self.I[i]])
            penalty += self.hard_penalty * sol.plan[i,Ic,:].sum()

        return penalty
    
    def constraint8(self, sol):
        # Each field must have some crop

        penalty = 0
        _sum = 0

        for k in range(self.K):
            _sum += sol.plan[:-1,:,k].sum() == 0
        
        penalty = self.hard_penalty * _sum

        return penalty


    @property
    def best_objective(self):
        return self.objective(self.best_sol)

    def init(self):

        self.population = []

        for _ in range(self.population_size):
            member = CRSPSolution(
                self.N,
                self.M,
                self.K,
                self.T,
                self.I,
                self.B
            )

            member.initialize()

            self.population.append(member)

    def fitness(self, sol):

        _fitness = self.objective(sol) +\
                   self.constraint2(sol) +\
                   self.constraint3(sol) +\
                   self.constraint4(sol) +\
                   self.constraint5(sol) +\
                   self.constraint6(sol) +\
                   self.constraint7(sol) +\
                   self.constraint8(sol)
        
        if sol.plan[:-1,:,:].sum() == 0:
            return float("-inf")
        

        if _fitness > self.best_fitness:
            self.best_fitness = _fitness
            self.best_sol = sol
        
        self.running_counter += 1
        self.running_sum += _fitness

        if self.running_counter%self.report_freq == 0:
            print("Average reward: ", self.running_sum/self.report_freq)
            self.running_counter = 0
            self.running_sum = 0
        
        return _fitness

    
    def solve(self, numof_generations):
        self.init()

        for gen_idx in range(numof_generations):

            scores = sorted([(self.fitness(s), s) for s in self.population], key = lambda x: x[0], reverse=True)
            #new_pop = [x[1] for x in scores[:2]]
            new_pop = [s[1] for s in scores[:self.elitism]]

            #for _ in range(self.population_size - 2):
            for _ in range(self.elitism,self.population_size):

                parent_idx, parent_idx_2 = rd.choices( range(self.numof_parents), k = 2)

                parent = scores[parent_idx][1]
                parent2 = scores[parent_idx_2][1]

                child = parent.crossover(parent2).mutation()
                #child = parent.crossover(parent2)
                new_pop.append(child)
            
            self.population = new_pop
    

    def render_solution(self, sol = None, save_fig = False):

        if sol == None:
            sol = self.best_sol
        
        labels = set()
        width = 1
        max_x = self.M
        fig, ax = plt.subplots()

        crop_colors = {}

        #plt.clf()

        for c in self.crops:
            crop_name = c["name"]
            crop_hash = abs(hash(crop_name))

            if crop_name == "fallow":
                crop_colors[crop_name] = (0.5,0.5,0.5)
            else:
                crop_colors[crop_name] = (
                    (crop_hash/10)%10/10,
                    (crop_hash/100)%10/10,
                    (crop_hash/1000)%10/10
                )

        for k in range(self.K):
            for period in range(self.M):
                crop = np.argmax(sol.plan[:,period,k])

                if sol.plan[:-1,period,k].sum() == 0: # nothing planted
                    continue
                    #crop = self.N-1 # fallow

                crop_name = self.crops[crop]["name"]
                xs = list(range(period, period + self.T[crop]))
                ys = [1] * len(xs)
                color = crop_colors[crop_name]
                bottom = [k] * len(xs)
                label = crop_name if crop_name not in labels else None
                labels.add(crop_name)
                ax.bar(xs, ys, width, bottom = bottom, color = color, label = label, align="edge")
        

        plt.title("Crop Rotation")
        ax.set_xlabel("Period")
        ax.set_ylabel("Plot")
        ax.set_xticks(list(range(max_x+1)))
        ax.set_yticks(list(range(1, self.K + 1)))

        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width*0.7, box.height])
        ax.legend(loc = 'upper center', bbox_to_anchor=(1.3,1), shadow = True)

        plt.grid(axis='x')
        plt.grid(axis='y')

        plt.show()
