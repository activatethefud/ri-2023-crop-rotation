import crsp
import pickle

problem0 = crsp.CRSP("data/problem0.json","obj_different_crops")
problem1 = crsp.CRSP("data/problem1.json","obj_different_crops")
problem2 = crsp.CRSP("data/problem2.json","obj_different_crops")
problem3 = crsp.CRSP("data/problem3.json","obj_different_crops")

sol = pickle.load(open("sol.pkl",'rb'))

#problem.solve()