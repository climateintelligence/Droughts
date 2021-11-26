import argparse
import glob
import os
import pickle
import numpy as np

from feature_selection import backwardFeatureSelection
from training import computeAccuracy

DELTA_GRID = [0.05, 0.25, 0.5, 1.0, 2.0]

def getThreshold(task, target, delta):
    if task == 1: # classification task
        return (delta**2)/2
    else:
        return delta/2*np.max(target)**2 # l-infinity norm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int) # k-neighbors
    parser.add_argument("--backward", type=str, default="t")
    parser.add_argument("--classification", nargs='?', const=1, type=int) 
    args = parser.parse_args()

    input_folder = os.path.join(os.getcwd(), "data", "real")
    filenames = glob.glob(os.path.join(input_folder, "*.pickle")) # get dataset files in pickle format

    # loading the first dataset in the folder
    dataset_file = open(filenames[0], "rb") 
    dataset = pickle.load(dataset_file) 
    dataset_file.close() 

    # --- FEATURE SELECTION ---
    features = dataset["X"]
    target = dataset["Y"]

    grid = DELTA_GRID
    grid.sort()

    if args.k is None:
        args.k=len(features)//20 # if not specified, fixed fraction of samples

    task = 1 if args.classification == 1 else 0 # task=1 -> classification

    res = {
        "delta" : [],
        "numSelected" : [],
        "accuracy" : []
    }

    for delta in grid:
        res["delta"].append(delta)
        threshold = getThreshold(task, target, delta) # soglia per il valore attuale di delta
        relevantFeatures = backwardFeatureSelection(delta,threshold,features,target,res, args.k) # basato su CMI
        res["accuracy"].append(computeAccuracy(task, relevantFeatures, target)) # performance di un modello lineare 
    