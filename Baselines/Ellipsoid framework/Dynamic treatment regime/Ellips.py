#######################################################################################################################
#   file name: Ellips
#
#   description:
#   this file builds a medical environment from the MIMICIII normalized data.
#   then runs the ellipsoid method to solve the linear COIRL problem.
#######################################################################################################################
# imports:
#######################################################################################################################
import sys
sys.path.append('../../../include')
import os
import numpy as np
from ICMDP import *
from domains import *
import random
import pickle
from Ellipsoid import *
from accuracy import *

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
# HYPER PARAMS
#######################################################################################################################
gamma=0.9
iters = 1000
epsilon = 5e-5
repeats = 10
tol = 1e-3
test_size = 300
RUN_TEST = True

#######################################################################################################################
# construct the CMDP
#######################################################################################################################
mdp = build_domain('MED',False)

# load real W:
real_W = np.load("../../../data/dynamic_treatment/offline/realW.npy")

# load test and train sets of contexts and initial states:
testset = np.load("../../../data/dynamic_treatment/all/testset.npy")[:test_size]
test_init_states = np.load("../../../data/dynamic_treatment/all/test_init_states.npy")[:test_size]
train_contexts = np.load("../../../data/dynamic_treatment/all/trainset.npy")
train_init_states = np.load("../../../data/dynamic_treatment/all/train_init_states.npy")

real_W /= np.linalg.norm(np.reshape(real_W,real_W.size),np.inf)

#######################################################################################################################
# Test the ellipsoid method in the environment
#######################################################################################################################
d={}

# Evaluate expert on test set:
if RUN_TEST:
    policies_expert = []
    test_expert_value = []
    for jk in range(len(test_init_states)):
        features_expert = mdp.solve_CMDP(gamma=gamma, tol=tol, W=real_W, context=testset[jk], flag='init',init_state=test_init_states[jk])
        policies_expert.append(features_expert.policy)
        value_expert = ((1-gamma)/real_W.shape[1]) * testset[jk] @ real_W @ features_expert.M
        test_expert_value.append(value_expert)
    test_expert_value = np.asarray(test_expert_value).mean()
    d["test_value"] = test_expert_value
train_rand_inds = np.arange(len(train_init_states))

# run seeds:
for trainset in range(repeats):
    save_obj(d, "ellips_values"+str(valuesindex))

    # random subset of the trainset:
    random.shuffle(train_rand_inds)
    Conts = train_contexts[train_rand_inds].copy()
    Conts = Conts[:iters]

    train_inits = train_init_states[train_rand_inds].copy()
    train_inits = train_inits[:iters]

    expert_values = np.zeros(len(Conts))
    agent_values = np.zeros(len(Conts))
    cumm_regret = np.zeros(len(Conts))
    contexts_seen = np.zeros(len(Conts))
    context_count = 0
    ellipsoid_volume = np.zeros(len(Conts))
    ERR = False

    E = Ellipsoid(real_W.size)

    max_updates = 2*real_W.size*(real_W.size + 1)*np.log(4*np.sqrt(real_W.size)/epsilon)
    min_ellipsoid_vol = (epsilon/2)**real_W.size
    num_seen = 0
    accuracy = []

    for t in range(iters):
        print("test ",trainset," timestep ",t, " Contexts seen:",context_count)

        # Agent and teacher play:
        Wt = E.getc().reshape(real_W.shape)
        features_expert = mdp.solve_CMDP(gamma=gamma, tol=tol, W=real_W,context=Conts[t],flag='init',init_state=train_inits[t])
        features_agent = mdp.solve_CMDP(gamma=gamma, tol=tol,  W=Wt, context=Conts[t],flag='init',init_state=train_inits[t])
        value_expert = ((1-gamma)/real_W.shape[1]) * Conts[t] @ real_W @ features_expert.M
        value_agent = ((1-gamma)/real_W.shape[1]) * Conts[t] @ real_W @ features_agent.M

        # Sanity checks:
        if t > 0 and ellipsoid_volume[t] > ellipsoid_volume[t-1]:
            print("ERR: Volume increased. t=",t," trainset=",trainset)
            ERR = True
        if context_count > max_updates:
            print("ERR: Went over max updates!. t=",t," trainset=",trainset)
            ERR = True
        if E.volume() < min_ellipsoid_vol:
            print("ERR: Ellipsoid too small!. t=",t," trainset=",trainset)
            ERR = True
        if not E.inside(real_W.reshape(real_W.size)):
            print("ERR: Real solution no longer in ellipsoid. t=",t," trainset=",trainset)
            ERR = True

        # Record results:
        expert_values[t] = value_expert
        agent_values[t] = value_agent
        contexts_seen[t] = context_count
        ellipsoid_volume[t] = E.volume()
        if t>0:
            cumm_regret[t] = cumm_regret[t-1] + value_expert - value_agent
        elif t==0:
            cumm_regret[t] = value_expert - value_agent

        # Calculate values on test set:
        # run test once every 10 contexts:
        if ( t > 0 and contexts_seen[t] != contexts_seen[t-1]):
            num_seen += 1

        if (RUN_TEST and num_seen >= 10) or (RUN_TEST and t==0):
            num_seen = 0
            test_agent_value = []
            accur = np.zeros(len(test_init_states))

            for jk in range(len(test_init_states)):
                features_agent = mdp.solve_CMDP(gamma=gamma, tol=tol, W=Wt, context=testset[jk], flag='init',init_state=test_init_states[jk])
                value_agent = ((1-gamma)/real_W.shape[1]) * testset[jk] @ real_W @ features_agent.M
                test_agent_value.append(value_agent)
                accur[jk] = accuracy_mesure(features_agent.policy,policies_expert[jk])

            accuracy.append(accur.mean())

            test_agent_value = np.asarray(test_agent_value).mean()
            print("test expert value: ",test_expert_value)
            print("test agent value: ", test_agent_value)
            d[trainset,"test_value",contexts_seen[t]] = test_agent_value

        # If agent is more than epsilon suboptimal, update ellipsoid:
        if value_expert - value_agent > epsilon:
            E.update(np.outer(Conts[t],features_expert.M-features_agent.M).reshape(real_W.size))
            context_count += 1


    # save data:
    np.save("obj/accuracy_ellips" + str(trainset) + ".npy",np.asarray(accuracy))
    d[trainset,"expert_values"] = expert_values
    d[trainset,"agent_values"] = agent_values
    d[trainset,"contexts_seen"] = contexts_seen
    d[trainset,"cumm_regret"] = cumm_regret

save_obj(d, "ellips_values"+str(valuesindex))
