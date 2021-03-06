#######################################################################################################################
#   file name: Ellips_non_linear
#
#   description:
#   this file builds a driving simulator environment.
#   then runs the ellipsoid method to solve the non-linear COIRL problem.
#######################################################################################################################
# imports:
#######################################################################################################################
import sys
sys.path.append('../../../include')
import numpy as np
from ICMDP import *
from domains import *
import os
import pickle
from Ellipsoid import *
from NL_functions import *
#######################################################################################################################
# data savings
#######################################################################################################################
valuesindex = 0
if not os.path.exists('obj'):
    os.mkdir('obj')

def save_obj(obj, name):
    with open('obj/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)
#######################################################################################################################
#HYPER PARAMS
#######################################################################################################################
gamma = 0.9
iters = 500
epsilon = 1e-3
repeats = 10
tol = 1e-4
test_size = 80
RUN_TEST = True
#######################################################################################################################
# construct the CMDP
#######################################################################################################################
mdp = build_domain('DRV',0)

# load test set of contexts:
testset = np.load("../../../data/autonomous_driving/all/test_set.npy")[:test_size]

#######################################################################################################################
# Test the Ellipsoid method in the environment
#######################################################################################################################
d={}

test_expert_value = 0

# Evaluate expert on test set
if RUN_TEST:
    test_expert_value = np.asarray([((1-gamma)/3) * NL(c) @ feat_exp(NL(c),mdp,tol) for c in testset]).mean()
    d["test_value"] = test_expert_value
    print("Expert test value: ",str(test_expert_value))

# run seeds:
for trainset in range(repeats):
    if trainset>0:
        save_obj(d, "ellips_nl_values"+str(valuesindex))

    # load train set of contexts:
    Conts = np.load("../../../data/autonomous_driving/all/train_set_" + str(trainset) + ".npy")

    expert_values = np.zeros(len(Conts))
    agent_values = np.zeros(len(Conts))
    cum_regret = np.zeros(len(Conts))
    contexts_seen = np.zeros(len(Conts))
    context_count = 0
    
    E = Ellipsoid(9)

    for t in range(iters):
        print("test ",trainset," timestep ",t, " Contexts seen:",str(context_count))

        # Agent and teacher play:
        ct = Conts[t]
        r = NL(ct)
        Wt = E.getc().reshape(3,3)
        r_est = ct @ Wt
        features_expert = feat_exp(r,mdp,tol)
        features_agent = feat_exp(r_est,mdp,tol)
        value_expert = ((1-gamma)/3) * r @ features_expert
        value_agent = ((1-gamma)/3) * r @ features_agent
        
        # Record results:
        expert_values[t] = value_expert
        agent_values[t] = value_agent
        contexts_seen[t] = context_count
        if t>0:
            cum_regret[t] = cum_regret[t-1] + value_expert - value_agent
        elif t==0:
            cum_regret[t] = value_expert - value_agent
        
        # Calculate values on test set:
        if (RUN_TEST and t>0 and contexts_seen[t] % 1 == 0 and contexts_seen[t] != contexts_seen[t-1]) or (RUN_TEST and t==0):
            test_agent_value = np.asarray([((1-gamma)/3) * NL(c) @ feat_exp(c @ Wt,mdp,tol) for c in testset]).mean()
            d[trainset,"test_value",contexts_seen[t]] = test_agent_value
            print(" == Generalization for ",str(contexts_seen[t]), "contexts: ",str(test_agent_value))

        # If agent is more than epsilon suboptimal, update ellipsoid:
        print("Value expert: ",str(value_expert)," Value agent: ",str(value_agent))
        if value_expert - value_agent > epsilon:
            E.update(np.outer(ct,features_expert-features_agent).reshape(9))
            context_count += 1

    # save data:
    d[trainset,"expert_values"] = expert_values
    d[trainset,"agent_values"] = agent_values
    d[trainset,"contexts_seen"] = contexts_seen
    d[trainset,"cum_regret"] = cum_regret

save_obj(d, "ellips_nl_values"+str(valuesindex))