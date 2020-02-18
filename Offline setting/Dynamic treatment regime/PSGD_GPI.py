#######################################################################################################################
#   file name: PSGD_GPI
#
#   description:
#   this file builds a medical environment from the MIMICIII normalized data.
#   then runs PSGD to solve the linear COIRL problem.
#   evaluates GPI policy.
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
tol = 1e-3
test_size = 300
RUN_TEST = True
max_iters = 60
trajectory_length = 40
batch_size = 10
context_nums = np.asarray([5,10,20,40,60,80,100])

#######################################################################################################################
# construct the CMDP
#######################################################################################################################

mdp = build_domain('MED',False)

# load real W:
real_W = np.load("../../data/dynamic_treatment/offline/realW.npy")

# load test and train sets of contexts and initial states:
testset = np.load("../../data/dynamic_treatment/all/testset.npy")[:test_size]
test_init_states = np.load("../../data/dynamic_treatment/all/test_init_states.npy")[:test_size]
train_contexts = np.load("../../data/dynamic_treatment/all/trainset.npy")
train_init_states = np.load("../../data/dynamic_treatment/all/train_init_states.npy")

#######################################################################################################################
# define functions for PSGD
#######################################################################################################################

def feat_exp(r,init_state):
    return mdp.solve_MDP(gamma=0.9,tol=tol,w=r,flag = 'init',init_state=init_state)

def NL(x):
    return x @ real_W

#######################################################################################################################
# Test the PSGD & GPI methods in the environment
#######################################################################################################################
d={}
d_gpi={}

training_contexts = []
training_featuresss = []
training_inits = []
train_expert_valuess = []
train_expert_policy = []
for ind in range(len(train_init_states)):
    print(ind)
    training_contexts.append(train_contexts[ind])
    tsoll = mdp.solve_CMDP(gamma=gamma, tol=tol, W=real_W / np.linalg.norm(real_W.flatten(), 2), context=train_contexts[ind],
                   flag='init', init_state=train_init_states[ind], length=trajectory_length)
    training_featuresss.append(tsoll.M)
    train_expert_policy.append(tsoll.policy)
    train_expert_valuess.append(train_contexts[ind] @ real_W @ tsoll.state_feature_exp[train_init_states[ind],:])


if RUN_TEST:
    policies_expert = []
    test_expert_value = []
    features_expert_z = []
    for jk in range(len(test_init_states)):
        features_expert = mdp.solve_CMDP(gamma=gamma, tol=tol, W=real_W/np.linalg.norm(real_W.flatten(),2), context=testset[jk], flag='init',init_state=test_init_states[jk])
        policies_expert.append(features_expert.policy)
        features_expert_z.append(features_expert.M)
        value_expert = testset[jk] @ real_W @ features_expert.M
        test_expert_value.append(value_expert)
    test_expert_value = np.asarray(test_expert_value).mean()
    d["test_value"] = test_expert_value
    d_gpi["test_value"] = test_expert_value
    print("Expert test value: ",str(test_expert_value))

d["train_value_expert"] = []
d_gpi["train_value_expert"] = []

# run seeds:
train_rand_inds = np.arange(len(train_init_states))
for trainset in range(repeats):
    if trainset > 0:
        save_obj(d, "coirl_values"+str(valuesindex))
        save_obj(d_gpi, "gpi_values"+str(valuesindex))

    W_init = 2 * np.random.rand(real_W.shape[0],real_W.shape[1]) - 1
    W_init /= np.linalg.norm(W_init.flatten(), 2)
    random.shuffle(train_rand_inds)
    Conts = train_contexts[train_rand_inds].copy()

    Train_features = np.asarray(training_featuresss)[train_rand_inds].copy()
    Train_policies = np.asarray(train_expert_policy)[train_rand_inds].copy()
    Train_values = np.asarray(train_expert_valuess)[train_rand_inds].copy()
    train_inits = np.asarray(train_init_states)[train_rand_inds].copy()

    accuracy = []
    action_dist = []
    
    W = W_init.copy()
    W_mean = W.copy()

    for num_contexts in context_nums:
        contexts = Conts[:num_contexts]
        training_inits = train_inits[:num_contexts]
        train_expert_value = np.asarray(Train_values[:num_contexts]).mean()
        print("train expert value: ",train_expert_value)
        training_features = Train_features[:num_contexts]
        training_policy = Train_policies[:num_contexts]
        d["train_value_expert"].append(train_expert_value)
        d_gpi["train_value_expert"].append(train_expert_value)
        W_sol = []
        for iter in range(max_iters):

            rand_sample = random.choices(list(range(num_contexts)), k=batch_size)
            r_est = contexts[rand_sample[0]] @ W / np.linalg.norm(W.flatten(), np.inf)
            features_agent = mdp.solve_MDP(0.9,tol=tol,w=r_est,flag='init',init_state=training_inits[rand_sample[0]])
            batch_outer = np.outer(contexts[rand_sample[0]], features_agent.M - training_features[rand_sample[0]])
            for sample in rand_sample[1:]:
                r_est = contexts[sample] @ W / np.linalg.norm(W.flatten(), np.inf)
                features_agent = mdp.solve_MDP(0.9, tol=tol, w=r_est, flag='init',init_state=training_inits[sample])
                batch_outer += np.outer(contexts[sample], features_agent.M - training_features[sample])

            batch_outer /= batch_size
            W = W - 0.25 * (0.95**iter) * batch_outer
            W /= max((np.linalg.norm(np.reshape(W, W.size), 2)), 1.0)
            W_mean = (W_mean * iter + W) / (iter + 1)

        W_sol.append(W_mean)
        train_agent_value_ar = []
        train_agent_accuracy_ar = []
        train_Q = []
        for cont in range(len(contexts)):
            solution = mdp.solve_MDP(0.9, tol=tol, w=contexts[cont] @ W_sol[0], flag='init',init_state=training_inits[cont])
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
            solution = mdp.feat_from_policy(0.9, contexts[cont], policy_i, tol=tol, flag='init',init_state=training_inits[cont], deep_policy=False)
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
            solution = mdp.solve_MDP(0.9,tol=tol,w=testset[cont] @ W_sol[0],flag='init',init_state=test_init_states[cont])
            policy_expert_i = policies_expert[cont]
            test_agent_value_ar.append(testset[cont] @ real_W @ solution.M)
            test_agent_accuracy_ar.append(100.0 * np.asarray(np.sum(policy_expert_i == solution.policy) / float(len(policy_expert_i))).mean())


            policy_i_gpi = mdp.cgpi(W_sol[0], testset[cont], train_Q)
            solution_gpi = mdp.feat_from_policy(0.9, testset[cont], policy_i_gpi, tol=tol, flag='init',init_state=test_init_states[cont], deep_policy=False)
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
