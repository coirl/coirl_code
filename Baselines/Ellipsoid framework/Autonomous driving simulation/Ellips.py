#######################################################################################################################
#   file name: Ellips
#
#   description:
#   this file builds a driving simulator environment
#   then runs a ellipsoid method to solve the linear COIRL problem.
#######################################################################################################################
# imports:
#######################################################################################################################
import sys
sys.path.append('../../../include')
import os
import numpy as np
from ICMDP import *
from domains import *
import pickle
from Ellipsoid import *
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
iters = 500
epsilon = 1e-3
gamma = 0.9
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

# load real W:
real_W = np.load("../../../data/autonomous_driving/offline/realW.npy")

#######################################################################################################################
# Test the ellipsoid method in the environment
#######################################################################################################################
d = {}

test_expert_value = 0

# Evaluate expert on test set:
if RUN_TEST:
    test_expert_value = []
    for c in testset:
        mdp.set_C(c)
        mdp.set_W(real_W)
        features_expert = mdp.solve_CMDP(gamma=gamma, tol=tol, flag='init')
        value_expert = ((1 - gamma) / real_W.shape[1]) * c @ real_W @ features_expert.M
        test_expert_value.append(value_expert)
    test_expert_value = np.asarray(test_expert_value).mean()
    d["test_value"] = test_expert_value

for trainset in range(repeats):
    if trainset > 0 and trainset % 5 == 1:
        save_obj(d, "ellips_values"+str(valuesindex))

    Conts = np.load("../../../data/autonomous_driving/all/train_set_" + str(trainset) + ".npy")
    Conts = Conts[:iters]
    expert_values = np.zeros(len(Conts))
    agent_values = np.zeros(len(Conts))
    cumm_regret = np.zeros(len(Conts))
    contexts_seen = np.zeros(len(Conts))
    context_count = 0
    ellipsoid_volume = np.zeros(len(Conts))
    ERR = False

    E = Ellipsoid(real_W.size)

    max_updates = 2 * real_W.size * (real_W.size + 1) * np.log(4 * np.sqrt(real_W.size) / epsilon)
    min_ellipsoid_vol = (epsilon / 2) ** real_W.size

    for t in range(iters):
        print("test ",trainset," timestep ",t, " Contexts seen:",context_count)

        # Agent and teacher play:
        ct = Conts[t]
        mdp.set_C(ct)
        Wt = E.getc().reshape(real_W.shape)
        mdp.set_W(real_W)
        features_expert = mdp.solve_CMDP(gamma=gamma, tol=tol, flag='init')
        mdp.set_W(Wt)
        features_agent = mdp.solve_CMDP(gamma=gamma, tol=tol, flag='init')
        value_expert = ((1 - gamma) / real_W.shape[1]) * ct @ real_W @ features_expert.M
        value_agent = ((1 - gamma) / real_W.shape[1]) * ct @ real_W @ features_agent.M

        # Sanity checks:
        if t > 0 and ellipsoid_volume[t] > ellipsoid_volume[t - 1]:
            print("ERR: Volume increased. t=", t, " trainset=", trainset)
            ERR = True
        if context_count > max_updates:
            print("ERR: Went over max updates!. t=", t, " trainset=", trainset)
            ERR = True
        if E.volume() < min_ellipsoid_vol:
            print("ERR: Ellipsoid too small!. t=", t, " trainset=", trainset)
            ERR = True
        if not E.inside(real_W.reshape(real_W.size)):
            print("ERR: Real solution no longer in ellipsoid. t=", t, " trainset=", trainset)
            ERR = True

        # Record results:
        expert_values[t] = value_expert
        agent_values[t] = value_agent
        contexts_seen[t] = context_count
        ellipsoid_volume[t] = E.volume()
        if t > 0:
            cumm_regret[t] = cumm_regret[t - 1] + value_expert - value_agent
        elif t == 0:
            cumm_regret[t] = value_expert - value_agent

        # Calculate values on test set:
        if (RUN_TEST and t > 0 and contexts_seen[t] % 5 == 0 and contexts_seen[t] != contexts_seen[t - 1]) or (
                RUN_TEST and t == 0):
            test_agent_value = []
            for c in testset:
                mdp.set_C(c)
                mdp.set_W(Wt)
                features_agent = mdp.solve_CMDP(gamma=gamma, tol=tol, flag='init')
                value_agent = ((1 - gamma) / real_W.shape[1]) * c @ real_W @ features_agent.M
                test_agent_value.append(value_agent)
            test_agent_value = np.asarray(test_agent_value).mean()
            d[trainset, "test_value", contexts_seen[t]] = test_agent_value

        # If agent is more than epsilon suboptimal, update ellipsoid:
        if value_expert - value_agent > epsilon:
            E.update(np.outer(ct, features_expert.M - features_agent.M).reshape(real_W.size))
            context_count += 1

    # save data:
    d[trainset, "expert_values"] = expert_values
    d[trainset, "agent_values"] = agent_values
    d[trainset, "contexts_seen"] = contexts_seen
    d[trainset, "cum_regret"] = cumm_regret

    if(ERR):
        print("Error during run ")

save_obj(d, "ellips_values"+str(valuesindex))
