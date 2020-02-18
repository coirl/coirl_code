#######################################################################################################################
#   file name: EW
#
#   description:
#   this file builds a medical environment from the MIMICIII normalized data.
#   then runs EW to solve the linear COIRL problem.
#######################################################################################################################
# imports:
#######################################################################################################################
import sys
sys.path.append('../../include')
import numpy as np
from ICMDP import *
from domains import *
import os
import random
import pickle
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
trajectory_length = 40
iters = 1000
repeats = 5
tol = 1e-3
batch_size = 1
test_size = 300
RUN_TEST = True

#######################################################################################################################
# construct the CMDP
#######################################################################################################################

mdp = build_domain('MED',True)

# load real W:
real_W = np.load("../../data/dynamic_treatment/online/realW.npy")

# load test and train sets of contexts and initial states:
testset = np.load("../../data/dynamic_treatment/all/testset.npy")[:test_size]
test_init_states = np.load("../../data/dynamic_treatment/all/test_init_states.npy")[:test_size]
train_contexts = np.load("../../data/dynamic_treatment/all/trainset.npy")
train_init_states = np.load("../../data/dynamic_treatment/all/train_init_states.npy")

#######################################################################################################################
# define functions for EW
#######################################################################################################################

def feat_exp(r,init_state):
    return mdp.solve_MDP(gamma=0.9,tol=tol,w=r,flag = 'init',init_state=init_state)

def NL(x):
    return x @ real_W

#######################################################################################################################
# Test the EW method in the environment
#######################################################################################################################
d={}

training_contexts_org = []
training_features_org = []
training_inits_org = []

for ind in range(len(train_init_states)):
    print(ind)
    training_contexts_org.append(train_contexts[ind])
    training_inits_org.append(train_init_states[ind])
    training_features_org.append(mdp.solve_CMDP(gamma=gamma, tol=tol, W=real_W/np.linalg.norm(real_W.flatten(),np.inf),context=train_contexts[ind],flag='init',init_state=train_init_states[ind],length=trajectory_length).M)

# Evaluate expert on test set:
if RUN_TEST:
    policies_expert = []
    test_expert_value = []
    features_expert_z = []
    for jk in range(len(test_init_states)):
        features_expert = mdp.solve_CMDP(gamma=gamma, tol=tol, W=real_W/np.linalg.norm(real_W.flatten(),2), context=testset[jk], flag='init',init_state=test_init_states[jk])
        policies_expert.append(features_expert.policy)
        features_expert_z.append(features_expert.M)
        value_expert = ((1-gamma)/real_W.shape[1]) * testset[jk] @ real_W @ features_expert.M
        test_expert_value.append(value_expert)
    test_expert_value = np.asarray(test_expert_value).mean()
    d["test_value"] = test_expert_value
    print("Expert test value: ",str(test_expert_value))

# run seeds:
train_rand_inds = np.arange(len(train_init_states))
for trainset in range(repeats):
    save_obj(d, "ew_values"+str(valuesindex))

    random.shuffle(train_rand_inds)

    # random subset of the trainset:
    training_contexts = np.asarray(training_contexts_org)[train_rand_inds].copy()
    training_features = np.asarray(training_features_org)[train_rand_inds].copy()
    training_inits = np.asarray(training_inits_org)[train_rand_inds].copy()

    expert_values = np.zeros(iters)
    agent_values = np.zeros(iters)
    contexts_seen = np.zeros(iters)

    accuracy = []

    # start from random W:
    W = np.load("../../data/dynamic_treatment/online/init_W_" + str(trainset) + ".npy")
    W_mean = W.copy()

    for t in range(iters+1):
        if (RUN_TEST and ((t % 5 == 0 and t < 500 ) or (t % 20 == 0))):
            test_agent_value = []
            test_agent_loss = []
            accur = np.zeros(len(test_init_states))
            act_dist = np.zeros(len(test_init_states))

            for jk in range(len(test_init_states)):
                r_est = testset[jk] @ W_mean/np.linalg.norm(W_mean.flatten(),2)
                features_agent_z = feat_exp(r_est, test_init_states[jk])
                loss_agent_z = ((1-gamma)/real_W.shape[1]) * testset[jk] @ (W_mean/np.linalg.norm(W_mean.flatten(),2)) @ (features_agent_z.M - features_expert_z[jk])
                value_agent_z = ((1-gamma)/real_W.shape[1]) * testset[jk] @ real_W @ (features_agent_z.M)
                test_agent_loss.append(abs(loss_agent_z))
                test_agent_value.append(abs(value_agent_z))
                accur[jk] = accuracy_mesure(features_agent_z.policy,policies_expert[jk])

            accuracy.append(accur.mean())

            test_agent_value = np.asarray(test_agent_value).mean()
            test_agent_loss = np.asarray(test_agent_loss).mean()
            d[trainset,"test_value",t] = test_agent_value
            d[trainset,"test_loss",t] = test_agent_loss
            print(" == Generalization for ",str(t), " iterations: \nValue: " + str(test_agent_value) + "\nLoss: " + str(test_agent_loss))
            print("Accuracy: ", accur.mean())

        rand_sample = np.arange(t*batch_size,(t+1)*batch_size)
        r_est = training_contexts[rand_sample[0]] @ W/np.linalg.norm(W.flatten(),2)
        features_agent = feat_exp(r_est,training_inits[rand_sample[0]])
        batch_outer = np.outer(training_contexts[rand_sample[0]], features_agent.M-training_features[rand_sample[0]])
        for sample in rand_sample[1:]:
            r_est = training_contexts[sample] @ W/np.linalg.norm(W.flatten(),2)
            features_agent = feat_exp(r_est,training_inits[sample])
            batch_outer += np.outer(training_contexts[sample], features_agent.M-training_features[sample])

        batch_outer /= batch_size

        W = np.multiply(W,np.exp(-1*10*(1-gamma) * np.sqrt(np.log((real_W.shape[0])*real_W.shape[1])/(2*(t + 1))) * batch_outer))
        if np.sum(np.reshape(W,W.size)) > 0:
            W /= np.sum(np.reshape(W,W.size))
        W_mean = (W_mean*t + W) / (t + 1)

    # save data:
    np.save("obj/accuracy_ew_" + str(trainset) + ".npy",np.asarray(accuracy))
    d[trainset,"expert_values"] = expert_values
    d[trainset,"agent_values"] = agent_values
    d[trainset,"contexts_seen"] = contexts_seen

save_obj(d, "ew_values"+str(valuesindex))
