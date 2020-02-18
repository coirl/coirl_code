#######################################################################################################################
#   file name: PSGD
#
#   description:
#   this file builds a driving simulator environment.
#   then runs PSGD method to solve the linear COIRL problem.
#######################################################################################################################
# imports:
#######################################################################################################################
import sys
sys.path.append('../../../include')
import numpy as np
import os
from ICMDP import *
from domains import *
import pickle

#######################################################################################################################
# data savings
#######################################################################################################################
valuesindex = 0
if not os.path.exists('obj'):
    os.mkdir('obj')


def save_obj(obj, name):
    with open('obj/' + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

#######################################################################################################################
# HYPER PARAMS
#######################################################################################################################
gamma = 0.9
repeats = 5
tol = 1e-4
test_size = 80
RUN_TEST = True
max_iters = 20
traj_len = 40
batch_size = 1
num_contexts = 40
traj_nums = np.asarray([5,10,20,30,40,60,80,100,1])

#######################################################################################################################
# construct the CMDP
#######################################################################################################################
mdp = build_domain('DRV',0)

# load test set of contexts:
testset = np.load("../../../data/autonomous_driving/all/test_set.npy")[:test_size]

# load real W:
real_W = np.load("../../../data/autonomous_driving/offline/realW.npy")

#######################################################################################################################
# Test the PSGD method in the environment
#######################################################################################################################
d = {}

test_expert_value = 0
test_expert_value_ar = []
policy_expert = []
for context in testset:
    feat_expert_i = mdp.solve_MDP(0.9, tol=tol, w=(context @ real_W), flag='init')
    policy_expert.append(feat_expert_i.policy)
    feat_expert = feat_expert_i.M
    test_expert_value_ar.append(context @ real_W @ feat_expert)
test_expert_value = np.asarray(test_expert_value_ar).mean()
d["test_value_expert"] = test_expert_value
print("Expert test value: ", str(test_expert_value))
d["train_value_expert"] = []

# run seeds:
for trainset in range(repeats):
    if trainset > 0:
        save_obj(d, "coirl_values" + str(valuesindex))
    run_i = trainset

    W_init = 2* np.random.rand(3,3) - 1
    W_init /= np.linalg.norm(W_init.flatten(),2)
    for traj_len in traj_nums:
        batch_size = int(num_contexts/2)
        contexts = np.load("../../../data/autonomous_driving/all/train_set_"+str(trainset)+".npy")[:num_contexts]

        training_features = []
        training_policy = []
        W_sol = []
        states_visited = []
        states_visited_av = 0
        train_expert_value_ar = []
        for ctt in contexts:
            soll = mdp.solve_MDP(0.9,tol=tol,w=(ctt @ real_W),flag='init',length=traj_len-1)
            training_policy.append(soll.policy)
            training_features.append(soll.M)
            train_expert_value_ar.append(ctt @ real_W @ soll.M)
            if traj_len != 1:
                states_visited.append(soll.ll)
                states_visited_av += soll.ll.size
        states_visited_av = float(states_visited_av) / num_contexts
        d[trainset,"states_visited",traj_len] = states_visited_av
        train_expert_value = np.asarray(train_expert_value_ar).mean()
        d["train_value_expert"].append(train_expert_value)
        W = W_init.copy()
        W_mean = W.copy()
        val_exp = 0

        for cont in range(len(contexts)):
            solution = mdp.solve_MDP(0.9, tol=tol, w=contexts[cont] @ real_W, flag='init')
            val_exp += contexts[cont] @ real_W @ (solution.M)
        val_exp /= float(len(contexts))
        ext_iter = 0
        for iter in range(40):
            if iter % 2 == 0:
                rand_sample = np.arange(batch_size)
            else:
                rand_sample = np.arange(batch_size,len(contexts))

            r_est = contexts[rand_sample[0]] @ W / np.linalg.norm(W.flatten(), np.inf)
            features_agent = mdp.solve_MDP(0.9,tol=tol,w=r_est,flag='init')
            batch_outer = np.outer(contexts[rand_sample[0]], features_agent.M - training_features[rand_sample[0]])
            for sample in rand_sample[1:]:
                r_est = contexts[sample] @ W / np.linalg.norm(W.flatten(), np.inf)
                features_agent = mdp.solve_MDP(0.9, tol=tol, w=r_est, flag='init')
                batch_outer += np.outer(contexts[sample], features_agent.M - training_features[sample])

            batch_outer /= batch_size
            W = W - 0.5 * (0.85**iter) * batch_outer
            W /= max((np.linalg.norm(np.reshape(W, W.size), 2)), 1.0)
            W_mean = (W_mean * iter + W) / (iter + 1)

        W_sol.append(W_mean)

        train_agent_value_ar = []
        train_agent_accuracy_ar = []

        for cont in range(len(contexts)):
            solution = mdp.solve_MDP(0.9, tol=tol, w=contexts[cont] @ W_sol[0], flag='init')
            curr_policy = training_policy[cont]
            train_agent_value_ar.append(contexts[cont] @ real_W @ solution.M)
            if traj_len == 1:
                train_agent_accuracy_ar.append(100.0 * np.asarray(np.sum(curr_policy == solution.policy) / float(len(curr_policy))).mean())
            else:
                train_agent_accuracy_ar.append(100.0 * np.asarray(np.sum(curr_policy[states_visited[cont]] == solution.policy[states_visited[cont]]) / float(len(curr_policy[states_visited[cont]]))).mean())
    
        train_agent_accuracy = np.asarray(train_agent_accuracy_ar).mean()
        train_agent_value = np.asarray(train_agent_value_ar).mean()
        print(" == Accuracy on train: ", str(train_agent_accuracy), "value on train: ", train_agent_value)
        d[trainset, "train_value", traj_len] = train_agent_value
        d[trainset, "train_accuracy", traj_len] = train_agent_accuracy

        save_obj(d, "coirl_values"+str(valuesindex))
        agent_vals = []
        runtimes = []
        agent_accurs = []
        test_agent_value_ar = []
        test_agent_accuracy_ar = []
        for cont in range(len(testset)):
            solution = mdp.solve_MDP(0.9,tol=tol,w=testset[cont] @ W_sol[0],flag='init')
            policy_expert_i = policy_expert[cont]
            test_agent_value_ar.append(testset[cont] @ real_W @ solution.M)
            test_agent_accuracy_ar.append(100.0 * np.asarray(np.sum(policy_expert_i == solution.policy) / float(len(policy_expert_i))).mean())

        test_agent_accuracy = np.asarray(test_agent_accuracy_ar).mean()
        agent_accurs.append(test_agent_accuracy)
        test_agent_value = np.asarray(test_agent_value_ar).mean()
        agent_vals.append(test_agent_value)
        print(" == Generalization for ",str(traj_len), "length: ", ", accuracy: ",str(test_agent_accuracy), " value: ",str(test_agent_value))
        d[trainset,"test_value",traj_len] = agent_vals
        d[trainset,"accuracy",traj_len] = agent_accurs

save_obj(d, "coirl_values"+str(valuesindex))
