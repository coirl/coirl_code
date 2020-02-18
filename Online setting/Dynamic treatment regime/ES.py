#######################################################################################################################
#   file name: ES
#
#   description:
#   this file builds a medical environment from the MIMICIII normalized data.
#   then runs ES to solve the linear COIRL problem.
#######################################################################################################################
# imports:
#######################################################################################################################
import sys
sys.path.append('../../include')
import os
import numpy as np
from ICMDP import *
from domains import *
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
# define functions for ES
#######################################################################################################################
def feat_exp(r,init_state):
    return mdp.solve_MDP(gamma=0.9,tol=tol,w=r,flag = 'init',init_state=init_state)

def NL(x):
    return x @ real_W
    
def NL_est(x,mat_list):
    y = x
    for i in range(len(mat_list)):
        y = y @ mat_list[i]/np.linalg.norm(mat_list[i].flatten(),np.inf)
    return y


def evaluate(map_eval, training_contexts, expert_feat_exp, agent_feat_exp):
    return np.asarray([(NL_est(training_contexts[j],map_eval) @ (agent_feat_exp[j] - expert_feat_exp[j])) \
                       for j in range(len(training_contexts))]).sum()


def Update_estimator(mat_list,training_contexts_all, expert_feat_exp_all, init_states_all, max_iter=500,
                     stepsize=0.05, decay=0.99, std=0.1, num_pts=20, epsilon=1e-3):
    # Create copy to not affect input matrices
    curr_list = [mat.copy() for mat in mat_list]

    # Initialize probability vector, make sure it is updated in 1st iteration
    # probs = np.zeros(len(training_contexts_all))

    iteration = 1
    print(max_iter)
    while iteration <= max_iter:
        agent_feat_exp_all = np.array([feat_exp(NL_est(training_contexts_all[j],curr_list), init_states_all[j]).M for j in range(len(training_contexts_all))])
        # early stopping condition
        if np.array([np.linalg.norm(agent_feat_exp_all[j]-expert_feat_exp_all[j],1) for j in range(len(expert_feat_exp_all))]).max() < epsilon:
            print("Early stop iteration ",iteration)
            return curr_list

        # Initialize step, calculate std for noise with decay
        training_contexts = training_contexts_all
        expert_feat_exp = expert_feat_exp_all
        agent_feat_exp = agent_feat_exp_all

        mat_num = len(curr_list)
        step = [np.zeros(shape=mat.shape) for mat in curr_list]
        for _ in range(int(num_pts/2)):
            # Create gaussian noises for matrices
            noise = [np.random.normal(size=mat.shape) for mat in curr_list]
            # Calculate step in both directions
            step_plus = [curr_list[j] + std*noise[j] for j in range(mat_num)]
            step_minus = [curr_list[j] - std*noise[j] for j in range(mat_num)]
            # Evaluate function and update step
            agent_feat_exp_all = np.array([feat_exp(NL_est(training_contexts_all[j],curr_list), init_states_all[j]).M for j in range(len(training_contexts_all))])
            eval_plus = evaluate(step_plus, training_contexts, expert_feat_exp, np.array([feat_exp(NL_est(training_contexts_all[j],step_plus), init_states_all[j]).M for j in range(len(training_contexts_all))]))
            eval_minus = evaluate(step_minus, training_contexts, expert_feat_exp,np.array([feat_exp(NL_est(training_contexts_all[j],step_minus), init_states_all[j]).M for j in range(len(training_contexts_all))]) )
            step = [step[j] + (eval_plus - eval_minus)*noise[j] for j in range(mat_num)]

        # Normalize step size
        step = [step[j] / np.linalg.norm(step[j].flatten()) for j in range(mat_num)]
        # Update current point
        curr_list = [curr_list[j] - stepsize*(decay**iteration)*step[j] for j in range(mat_num)]
        curr_list = [mat/np.linalg.norm(mat.flatten()) for mat in curr_list]
        # If point is worse on this minibatch, cancel this step
        iteration += 1

    return curr_list

#######################################################################################################################
# Test the ES method in the environment
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
    save_obj(d, "es_values"+str(valuesindex))

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
    weights_init = [np.load("../../data/dynamic_treatment/online/init_W_" + str(trainset) + ".npy")]
    weights = weights_init.copy()
    weights_mean = weights_init.copy()

    for t in range(iters+1):
        if (RUN_TEST and (t % 5 == 0 and t < 500) or (t % 20 == 0)):
            test_agent_value = []
            test_agent_loss = []
            accur = np.zeros(len(test_init_states))
            act_dist = np.zeros(len(test_init_states))

            for jk in range(len(test_init_states)):
                r_est = NL_est(testset[jk],weights_mean/np.linalg.norm(weights_mean[0].flatten(),2))
                features_agent = feat_exp(r_est, test_init_states[jk])
                loss_agent_t = ((1-gamma)/real_W.shape[1]) * testset[jk] @ (weights_mean[0]/np.linalg.norm(weights_mean[0].flatten(),2)) @ (features_agent.M - features_expert_z[jk])
                value_agent_t = ((1-gamma)/real_W.shape[1]) * testset[jk] @ real_W @ features_agent.M
                test_agent_value.append(abs(value_agent_t))
                test_agent_loss.append(abs(loss_agent_t))
                accur[jk] = accuracy_mesure(features_agent.policy,policies_expert[jk])

            accuracy.append(accur.mean())

            test_agent_value = np.asarray(test_agent_value).mean()
            test_agent_loss = np.asarray(test_agent_loss).mean()
            d[trainset,"test_value",t] = test_agent_value
            d[trainset,"test_loss",t] = test_agent_loss
            print(" == Generalization for ",str(t), " iterations: \nValue: " + str(test_agent_value) + "\nLoss: " + str(test_agent_loss))
            print("Accuracy: ", accur.mean())

        rand_batch = np.arange(t*batch_size,(t+1)*batch_size)
        weights = Update_estimator(weights, np.asarray(training_contexts)[rand_batch],
                                       np.asarray(training_features)[rand_batch],
                                       np.asarray(training_inits)[rand_batch],
                                       max_iter=1, stepsize=0.25, decay=0.95**t,
                                       std=0.1, num_pts=1000, epsilon=5e-4)
        weights_mean = [(weights_mean[0]*t + weights[0]) / (t + 1)]

    # save data:
    np.save("obj/accuracy_" + str(trainset) + ".npy",np.asarray(accuracy))
    d[trainset,"expert_values"] = expert_values
    d[trainset,"agent_values"] = agent_values
    d[trainset,"contexts_seen"] = contexts_seen

save_obj(d, "es_values"+str(valuesindex))
