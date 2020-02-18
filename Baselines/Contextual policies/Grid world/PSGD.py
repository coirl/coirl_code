#######################################################################################################################
#   file name: PSGD
#
#   description:
#   this file builds a grid-world environment.
#   then runs PSGD to solve the linear COIRL problem.
#######################################################################################################################
# imports:
#######################################################################################################################
import sys
sys.path.append('../../../include')
import numpy as np
from ICMDP import *
from domains import *
import pickle
import random
import os

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
gamma = 0.9
test_size = 100
max_iters = 1000
context_nums = np.asarray([100,200,300,400,500])
dims_vec = [(3,3),(4,4),(5,5),(6,6),(7,7),(8,8),(9,9),(10,10)]
repeats = 5
tol = 1e-5
RUN_TEST = True
batch_size = 12

#######################################################################################################################
# Train & Evaluation
#######################################################################################################################
for dims in dims_vec:
    print("####################################################################")
    print("%%%%%%%%%%%%% running with dimensions: " + str(dims[0]) + "X" + str(dims[1]))
    print("####################################################################")
    d = {}
    d["context_nums"] = np.asarray(context_nums)
    states = np.arange(dims[0]*dims[1])
    dim_contexts = len(states)

    all_contexts = np.load("../../../data/grid_world/rand_contexts_" + str(dims[0]) + "X" + str(dims[1]) + ".npy")
    test_contexts = all_contexts[all_contexts.shape[0] - test_size: all_contexts.shape[0]]

    mdp = build_domain('GRID', [dims[0], dims[1], False])

    real_W = np.eye(dim_contexts)
    test_expert_value = 0
    test_expert_value_ar = []
    policy_expert = []
    for context in test_contexts:
        feat_expert_i = mdp.solve_MDP(gamma, tol=tol, w=(context @ real_W), flag='uniform')
        policy_expert.append(feat_expert_i.policy)
        feat_expert = feat_expert_i.M
        test_expert_value_ar.append(context @ real_W @ feat_expert)
    test_expert_value = np.asarray(test_expert_value_ar).mean()
    d["test_value_expert"] = test_expert_value
    print("Expert test value: ", str(test_expert_value))

    d["train_value_expert"] = []
    for trainset in range(repeats):
        run_i = trainset
        for num_contexts in context_nums:
            contexts = all_contexts[trainset*context_nums[-1]:trainset*context_nums[-1] + num_contexts]
            training_features = []
            training_policy = []
            train_expert_value_ar = []
            real_W = np.eye(dim_contexts)
            for ctt in contexts:
                soll = mdp.solve_MDP(gamma,tol=tol,w=(ctt @ real_W),flag='uniform')
                training_policy.append(soll.policy)
                training_features.append(soll.M)
                train_expert_value_ar.append(ctt @ real_W @ soll.M)
            train_expert_value = np.asarray(train_expert_value_ar).mean()
            d["train_value_expert"].append(train_expert_value)
            W = np.ones([len(states), len(states)])
            W /= np.sum(W.flatten())
            W_mean = W.copy()
            for iter in range(max_iters):
                rand_sample = random.choices(list(range(num_contexts)), k=batch_size)
                r_est = contexts[rand_sample[0]] @ W / np.linalg.norm(W.flatten(), np.inf)
                features_agent = mdp.solve_MDP(gamma,tol=tol,w=r_est,flag='uniform')
                batch_outer = np.outer(contexts[rand_sample[0]], features_agent.M - training_features[rand_sample[0]])
                for sample in rand_sample[1:]:
                    r_est = contexts[sample] @ W / np.linalg.norm(W.flatten(), np.inf)
                    features_agent = mdp.solve_MDP(gamma, tol=tol, w=r_est, flag='uniform')
                    batch_outer += np.outer(contexts[sample], features_agent.M - training_features[sample])

                batch_outer /= batch_size

                W = W - 0.1/ np.sqrt(2 * dim_contexts * len(states) * (iter + 1)) * batch_outer
                W /= max((np.linalg.norm(np.reshape(W, W.size), np.inf)), 1.0)
                W_mean = (W_mean * iter + W) / (iter + 1)

            W_sol = [W_mean]
            train_agent_value_ar = []
            train_agent_accuracy_ar = []
            for cont in range(len(contexts)):
                solution = mdp.solve_MDP(gamma, tol=tol, w=contexts[cont] @ W_sol[0], flag='uniform')
                curr_policy = training_policy[cont]
                train_agent_value_ar.append(contexts[cont] @ real_W @ solution.M)
                train_agent_accuracy_ar.append(100.0 * np.asarray(np.sum(curr_policy == solution.policy) / float(len(curr_policy))).mean())

            train_agent_accuracy = np.asarray(train_agent_accuracy_ar).mean()
            train_agent_value = np.asarray(train_agent_value_ar).mean()
            print(" == Accuracy on train: ", str(train_agent_accuracy), "value on train: ", train_agent_value)
            d[trainset, "train_value", num_contexts] = train_agent_value
            d[trainset, "train_accuracy", num_contexts] = train_agent_accuracy

            save_obj(d, "coirl_values"+str(valuesindex))
            agent_vals = []
            agent_accurs = []
            test_agent_value_ar = []
            test_agent_accuracy_ar = []
            for cont in range(len(test_contexts)):
                solution = mdp.solve_MDP(gamma,tol=tol,w=test_contexts[cont] @ W_sol[0],flag='uniform')
                policy_expert_i = policy_expert[cont]
                test_agent_value_ar.append(test_contexts[cont] @ real_W @ solution.M)
                test_agent_accuracy_ar.append(100.0 * np.asarray(np.sum(policy_expert_i == solution.policy) / float(len(policy_expert_i))).mean())

            test_agent_accuracy = np.asarray(test_agent_accuracy_ar).mean()
            agent_accurs.append(test_agent_accuracy)
            test_agent_value = np.asarray(test_agent_value_ar).mean()
            agent_vals.append(test_agent_value)
            print(" == Generalization for ",str(num_contexts), "contexts: ", ", accuracy: ",str(test_agent_accuracy), " value: ",str(test_agent_value))
            d[trainset,"test_value",num_contexts] = agent_vals
            d[trainset,"accuracy",num_contexts] = agent_accurs

    save_obj(d, "coirl_values"+str(valuesindex))
    valuesindex += 1

