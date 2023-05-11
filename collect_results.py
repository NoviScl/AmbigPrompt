import os
import numpy as np
import csv

def get_file_names(directory):
    file_names = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_names.append(file)
    return file_names

def get_em(lines):
    counter = 0
    correct = 0
    for line in lines:
        if "raw_label_probs:" in line:
            counter += 1
            numbers = line.split()
            numbers = [float(numbers[1].strip()[1:]), float(numbers[2].strip()[:-1])]
            if numbers[1] > numbers[0]:
                correct += 1
    return correct / counter * 100

## list all files in the directory
directory = "logs_ambiguous"
file_names = get_file_names(directory)
# testsets = [fname.split("_testset")[0] for fname in file_names]

testsets = ["civilcomments_cluster3_vs_others"]
seeds = ["seed0", "seed10"]
models = ["text-davinci-002"]
subsets = ["testset_1_1", "testset_0_0", "testset_1_0", "testset_0_1"]
fieldnames = ["dataset", "model", "feature_pair", "intervention", "seed", "h1_ambig_acc", "h2_ambig_acc"]

data = [] ## gather results for plotting 
davinci_h1_ambig = []
davinci_h2_ambig = []
TD_h1_ambig = []
TD_h2_ambig = []

for testset in testsets:
    for model in models:
        h1_acc = []
        h2_acc = []
        h1_ambig_acc = []
        h2_ambig_acc = []
        for subset in subsets:
            EMs = []
            for seed in seeds:
                fname = '_'.join([testset, subset, model, seed])+'.log'
                with open("logs_ambiguous_unknown/"+fname, "r") as f:
                    lines = f.readlines()
                    ## take Calibrate-before-Use results by default
                    for line in lines:
                        if "EM: " in line and '=' in line and '%' in line:
                            em = line.split('=')[1].strip()
                            em = float(em[:-1]) 

                    ## take raw results without calibration
                    # em = get_em(lines)
                    
                    # print (fname)
                    EMs.append(em)

                    if subset == "testset_1_1":
                        h1_acc.append(em) 
                        h2_acc.append(em) 
                    elif subset == "testset_0_0":
                        h1_acc.append(100 - em)
                        h2_acc.append(100 - em)
                    elif subset == "testset_1_0":
                        h1_acc.append(em) 
                        h2_acc.append(100 - em)
                        h1_ambig_acc.append(em) 
                        h2_ambig_acc.append(100 - em)
                    elif subset == "testset_0_1":
                        h1_acc.append(100 - em)
                        h2_acc.append(em) 
                        h1_ambig_acc.append(100 - em)
                        h2_ambig_acc.append(em) 
            
            print ('_'.join([testset, subset, model]))
            print ("Frequency of prediction label 1 (as opposed to label 0)")
            print ("mean: ", np.mean(EMs))
            print ("std: ", np.std(EMs))
            print ()
        
        if model == "davinci":
            davinci_h1_ambig.append(np.mean(h1_ambig_acc))
            davinci_h2_ambig.append(np.mean(h2_ambig_acc))
            print ("Ambig Acc: ", model, "; H1 Acc: ", np.mean(h1_ambig_acc), "; H2 Acc: ", np.mean(h2_ambig_acc))
            print ()
        elif model == "text-davinci-002":
            TD_h1_ambig.append(np.mean(h1_ambig_acc))
            TD_h2_ambig.append(np.mean(h2_ambig_acc))
            print ("Ambig Acc: ", model, "; H1 Acc: ", np.mean(h1_ambig_acc), "; H2 Acc: ", np.mean(h2_ambig_acc))
            print ()
        
        # data.append({"dataset": testset, "model": model, "feature_pair": testset, "intervention": "none", "seed": 0, \
        #     "h1_ambig_acc": h1_ambig_acc[0], "h2_ambig_acc": h2_ambig_acc[0]})
        # data.append({"dataset": "BoolQ", "model": model, "feature_pair": testset, "intervention": "none", "seed": 40, \
        #     "h1_ambig_acc": h1_ambig_acc[1], "h2_ambig_acc": h2_ambig_acc[1]})


print ("H1 Davinci: ", np.mean(davinci_h1_ambig))
print ("H2 Davinci: ", np.mean(davinci_h2_ambig))
print ("H1 TD002: ", np.mean(TD_h1_ambig))
print ("H2 TD002: ", np.mean(TD_h2_ambig))


