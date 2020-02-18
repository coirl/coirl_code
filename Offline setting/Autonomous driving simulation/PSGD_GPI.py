#######################################################################################################################
#   file name: PSGD_GPI
#
#   description:
#   this file builds a driving simulator environment.
#   then runs a PSGD to solve the linear COIRL problem.
#   evaluates GPI policy.
#######################################################################################################################
# imports:
#######################################################################################################################
import sys
sys.path.append('../../include')
import numpy as np
import os
from ICMDP import *
from domains import *
import pickle
import random

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
max_iters = 40
traj_len = 40
batch_size = 1
context_nums = np.asarray([5,10,20,40,60,80])

#######################################################################################################################
# construct the CMDP
#######################################################################################################################
mdp = build_domain('DRV',0)

# load test set of contexts:
testset = np.load("../../data/autonomous_driving/all/test_set.npy")[:test_size]

# load real W:
real_W = np.load("../../data/autonomous_driving/offline/realW.npy")

#######################################################################################################################
# Test the PSGD & GPI methods in the environment
#######################################################################################################################
d = {}
d_gpi = {}

test_expert_value = 0
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
d_gpi["test_value_expert"] = test_expert_value
print("Expert test value: ", str(test_expert_value))

d["train_value_expert"] = []
d_gpi["train_value_expert"] = []

# run seeds:
for trainset in range(repeats):
    if trainset > 0:
        save_obj(d, "coirl_values" + str(valuesindex))
        save_obj(d_gpi, "gpi_values" + str(valuesindex))
    run_i = trainset

    W_init = 2* np.random.rand(3,3) - 1
    W_init /= np.linalg.norm(W_init.flatten(),2)
    batch_size = 10
    for num_contexts in context_nums:
        contexts = np.load("../../data/autonomous_driving/all/train_set_" + str(trainset) + ".npy")[:num_contexts]

        training_features = []
        training_policy = []
        W_sol = []
        train_expert_value_ar = []
        for ctt in contexts:
            soll = mdp.solve_MDP(0.9,tol=tol,w=(ctt @ real_W),flag='init',length=traj_len)
            training_policy.append(soll.policy)
            training_features.append(soll.M)
        W = W_init.copy()
        W_mean = W.copy()
        val_exp = 0
        for cont in range(len(contexts)):
            solution = mdp.solve_MDP(0.9, tol=tol, w=contexts[cont] @ real_W, flag='init')
            val_exp += contexts[cont] @ real_W @ (solution.M)
            train_expert_value_ar.append(contexts[cont] @ real_W @ solution.M)
        train_expert_value = np.asarray(train_expert_value_ar).mean()
        d["train_value_expert"].append(train_expert_value)
        d_gpi["train_value_expert"].append(train_expert_value)
        val_exp /= float(len(contexts))
        ext_iter = 0
        print("train expert value: ",val_exp)
        for iter in range(max_iters):
            rand_sample = random.choices(list(range(num_contexts)), k=batch_size)
            r_est = contexts[rand_sample[0]] @ W / np.linalg.norm(W.flatten(), np.inf)
            features_agent = mdp.solve_MDP(0.9,tol=tol,w=r_est,flag='init')
            batch_outer = np.outer(contexts[rand_sample[0]], features_agent.M - training_features[rand_sample[0]])
            for sample in rand_sample[1:]:
                r_est = contexts[sample] @ W / np.linalg.norm(W.flatten(), np.inf)
                features_agent = mdp.solve_MDP(0.9, tol=tol, w=r_est, flag='init')
                batch_outer += np.outer(contexts[sample], features_agent.M - training_features[sample])

            batch_outer /= batch_size
            W = W - 0.3 * (0.92**iter) * batch_outer
            W /= max((np.linalg.norm(np.reshape(W, W.size), 2)), 1.0)
            W_mean = (W_mean * iter + W) / (iter + 1)

        W_sol.append(W_mean)

        train_agent_value_ar = []
        train_agent_accuracy_ar = []
        train_Q = []
        for cont in range(len(contexts)):
            solution = mdp.solve_MDP(0.9, tol=tol, w=contexts[cont] @ W_sol[0], flag='init')
            curr_policy = training_policy[cont]
            train_Q.append(solution.Q)
            train_agent_value_ar.append(contexts[cont] @ real_W @ solution.M)
            train_agent_accuracy_ar.append(100.0 * np.asarray(np.sum(curr_policy == solution.policy) / float(len(curr_policy))).mean())

        train_agent_accuracy = np.asarray(train_agent_accuracy_ar).mean()
        train_agent_value = np.asarray(train_agent_value_ar).mean()
        print(" == Accuracy on train: ", str(train_agent_accuracy), "value on train: ", train_agent_value)
        d[trainset, "train_value", num_contexts] = train_agent_value
        d[trainset, "train_accuracy", num_contexts] = train_agent_accuracy

        train_agent_value_ar_gpi = []
        train_agent_accuracy_ar_gpi = []
        for cont in range(len(contexts)):
            policy_i = mdp.cgpi(W_sol[0], contexts[cont], train_Q)
            solution = mdp.feat_from_policy(0.9, contexts[cont], policy_i, tol=tol, flag='init', deep_policy=False)
            curr_policy = training_policy[cont]
            train_agent_value_ar_gpi.append(contexts[cont] @ real_W @ solution.M)
            train_agent_accuracy_ar_gpi.append(100.0 * np.asarray(
                [solution.policy[ind, curr_policy[ind]] / float(len(curr_policy)) for ind in
                 range(len(curr_policy))]).sum())

        train_agent_accuracy_gpi = np.asarray(train_agent_accuracy_ar_gpi).mean()
        train_agent_value_gpi = np.asarray(train_agent_value_ar_gpi).mean()


        print(" == Accuracy on train - gpi: ", str(train_agent_accuracy_gpi), "value on train: ", train_agent_value_gpi)
        d_gpi[trainset, "train_value", num_contexts] = train_agent_value_gpi
        d_gpi[trainset, "train_accuracy", num_contexts] = train_agent_accuracy_gpi


        save_obj(d, "coirl_values"+str(valuesindex))
        save_obj(d_gpi, "gpi_values" + str(valuesindex))
        agent_vals = []
        runtimes = []
        agent_accurs = []
        test_agent_value_ar = []
        test_agent_accuracy_ar = []

        agent_vals_gpi = []
        runtimes_gpi = []
        agent_accurs_gpi = []
        test_agent_value_ar_gpi = []
        test_agent_accuracy_ar_gpi = []

        for cont in range(len(testset)):
            solution = mdp.solve_MDP(0.9,tol=tol,w=testset[cont] @ W_sol[0],flag='init')
            policy_expert_i = policy_expert[cont]
            test_agent_value_ar.append(testset[cont] @ real_W @ solution.M)
            test_agent_accuracy_ar.append(100.0 * np.asarray(np.sum(policy_expert_i == solution.policy) / float(len(policy_expert_i))).mean())


            policy_i_gpi = mdp.cgpi(W_sol[0], testset[cont], train_Q)
            solution_gpi = mdp.feat_from_policy(0.9, testset[cont], policy_i_gpi, tol=tol, flag='init', deep_policy=False)
            test_agent_value_ar_gpi.append(testset[cont] @ real_W @ solution_gpi.M)
            test_agent_accuracy_ar_gpi.append(100.0 * np.asarray(
                [solution_gpi.policy[ind, policy_expert_i[ind]] / float(len(policy_expert_i)) for ind in
                 range(len(policy_expert_i))]).sum())

        test_agent_accuracy = np.asarray(test_agent_accuracy_ar).mean()
        agent_accurs.append(test_agent_accuracy)
        test_agent_value = np.asarray(test_agent_value_ar).mean()
        agent_vals.append(test_agent_value)


        test_agent_accuracy_gpi = np.asarray(test_agent_accuracy_ar_gpi).mean()
        agent_accurs_gpi.append(test_agent_accuracy_gpi)
        test_agent_value_gpi = np.asarray(test_agent_value_ar_gpi).mean()
        agent_vals_gpi.append(test_agent_value_gpi)

        print(" == Generalization for ",str(num_contexts), "contexts: ", ", accuracy: ",str(test_agent_accuracy), " value: ",str(test_agent_value))
        d[trainset,"test_value",num_contexts] = agent_vals
        d[trainset,"accuracy",num_contexts] = agent_accurs
        print(" == Generalization - gpi for ",str(num_contexts), "contexts: ", ", accuracy: ",str(test_agent_accuracy_gpi), " value: ",str(test_agent_value_gpi))
        d_gpi[trainset,"test_value",num_contexts] = agent_vals_gpi
        d_gpi[trainset,"accuracy",num_contexts] = agent_accurs_gpi


save_obj(d, "coirl_values"+str(valuesindex))
save_obj(d_gpi, "gpi_values"+str(valuesindex))


