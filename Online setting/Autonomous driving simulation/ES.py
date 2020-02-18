#######################################################################################################################
#   file name: ES
#
#   description:
#   this file builds a driving simulator environment.
#   then runs ES to solve the linear COIRL problem.
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
#HYPER PARAMS
#######################################################################################################################
gamma = 0.9
traj_len = 40
batch_size = 1
iters = 500
train_size =  1000
repeats = 5
tol = 1e-4
test_size = 80
RUN_TEST = True
#######################################################################################################################
# construct the CMDP
#######################################################################################################################
mdp = build_domain('DRV',0)

# load test set of contexts:
testset = np.load("../../data/autonomous_driving/all/test_set.npy")[:test_size]

# load real W:
real_W = np.load("../../data/autonomous_driving/online/realW.npy")

#######################################################################################################################
# define functions for ES
#######################################################################################################################

def feat_exp(r,policy=False):
    if policy:
        return mdp.solve_MDP(gamma=gamma,tol=tol,w=r,flag = 'init')

    return mdp.solve_MDP(gamma=gamma,tol=tol,w=r,flag = 'init').M

def NL(x):
    return x @ real_W/np.linalg.norm(real_W.flatten(),2)

def NL_est(x,mat_list):
    y = x
    for i in range(len(mat_list)):
        y = y @ mat_list[i]/np.linalg.norm(mat_list[i].flatten(),2)
    return y

def evaluate(map_eval, training_contexts, expert_feat_exp, agent_feat_exp):
    return np.asarray([(NL_est(training_contexts[j],map_eval) @ (agent_feat_exp[j] - expert_feat_exp[j])) \
                       for j in range(len(training_contexts))]).sum()

def evaluate_verbose(eval_map, training_contexts_all, expert_feat_exp_all):
    agent_r_est = [NL_est(training_contexts_all[j],eval_map) for j in range(len(training_contexts_all))]
    agent_feat_exp = [feat_exp(NL_est(training_contexts_all[j],eval_map)) for j in range(len(training_contexts_all))]
    value_err = np.asarray([(agent_r_est[j] @ (agent_feat_exp[j] - expert_feat_exp_all[j]))**2 \
                            for j in range(len(training_contexts_all))]).sum()
    feat_err  = np.asarray([np.linalg.norm(agent_feat_exp[j] - expert_feat_exp_all[j],2) \
                            for j in range(len(training_contexts_all))]).sum()
    dir_err = np.asarray([(agent_r_est[j] @ (agent_feat_exp[j] - expert_feat_exp_all[j])) \
                            for j in range(len(training_contexts_all))]).sum()
    print("value err: ",value_err, " feat_exp err: ",feat_err, " dir err: ",dir_err)

def Update_estimator(mat_list,training_contexts_all, expert_feat_exp_all, max_iter=500,
                     stepsize=0.05, decay=0.99, std=0.1, num_pts=20, batch_size=1, epsilon=1e-3):
    # Create copy to not affect input matrices
    curr_list = [mat.copy() for mat in mat_list]

    # Initialize probability vector, make sure it is updated in 1st iteration
    probs = np.zeros(len(training_contexts_all))

    iteration = 1
    while iteration <= max_iter:
        agent_feat_exp_all = np.array([feat_exp(NL_est(training_context,curr_list)) for training_context in training_contexts_all])
        # early stopping condition
        if np.array([np.linalg.norm(agent_feat_exp_all[j]-expert_feat_exp_all[j],1) for j in range(len(expert_feat_exp_all))]).max() < epsilon:
            print("Early stop iteration ",iteration)
            return curr_list
        
        probs = np.array([evaluate(curr_list, np.expand_dims(training_contexts_all[j],0),
                          np.expand_dims(expert_feat_exp_all[j],0), np.expand_dims(agent_feat_exp_all[j],0)) for j in range(len(training_contexts_all))])

        # Select minibatch for this iteration using the probability vector
        probs = np.maximum(probs,0)
        probs += 1e-4
        indices = np.random.choice(a=list(range(len(training_contexts_all))), p=probs/probs.sum(), size=batch_size, replace=False)
        training_contexts = training_contexts_all[indices]
        expert_feat_exp = expert_feat_exp_all[indices]
        agent_feat_exp = agent_feat_exp_all[indices]

        # Initialize step, calculate std for noise with decay
        mat_num = len(curr_list)
        step = [np.zeros(shape=mat.shape) for mat in curr_list]
        for _ in range(int(num_pts/2)):

            # Create gaussian noises for matrices
            noise = [np.random.normal(size=mat.shape) for mat in curr_list]

            # Calculate step in both directions
            step_plus = [curr_list[j] + std*noise[j] for j in range(mat_num)]
            step_minus = [curr_list[j] - std*noise[j] for j in range(mat_num)]
            step_plus = [np.maximum(mat,0) for mat in step_plus]
            step_minus = [np.maximum(mat,0) for mat in step_minus]
            step_plus = [mat/np.sum(mat.flatten()) for mat in step_plus]
            step_minus = [mat/np.sum(mat.flatten()) for mat in step_minus]

            # Evaluate function and update step
            eval_plus = evaluate(step_plus, training_contexts, expert_feat_exp, agent_feat_exp)
            eval_minus = evaluate(step_minus, training_contexts, expert_feat_exp, agent_feat_exp)
            step = [step[j] + (eval_plus - eval_minus)*noise[j] for j in range(mat_num)]

        # Normalize step size
        step = [step[j] /(num_pts*std)  for j in range(mat_num)]

        # Update current point
        curr_list = [curr_list[j] - stepsize*(decay**iteration)*step[j] for j in range(mat_num)]
        curr_list = [np.maximum(mat,0) for mat in curr_list]
        curr_list = [mat/np.sum(mat.flatten()) for mat in curr_list]

        # If point is worse on this minibatch, cancel this step
        iteration += 1

    return curr_list

#######################################################################################################################
# Test the ES method in the environment
#######################################################################################################################
d = {}
    
test_expert_value = 0
expert_feat_exp_z = []
test_expert_value_ar = []
policies_expert = []

# Evaluate expert on test set
if RUN_TEST:
    for jk in range(test_size):
        feat_expert = feat_exp(NL(testset[jk]),policy=True)
        expert_feat_exp_z.append(feat_expert.M)
        policies_expert.append(feat_expert.policy)
        test_expert_value_ar.append(((1-gamma)/3) * testset[jk] @ real_W @ feat_expert.M)
    test_expert_value = np.asarray(test_expert_value_ar).mean()
    d["test_value"] = test_expert_value
    print("Expert test value: ",str(test_expert_value))

# run seeds:
for trainset in range(repeats):
    if trainset>0:
        save_obj(d, "es_values"+str(valuesindex))

    # load train set of contexts:
    Conts = np.load("../../data/autonomous_driving/all/train_set_"+str(trainset)+".npy")[:train_size]
    training_contexts = []
    training_features = []
    for cc in Conts:
        training_contexts.append(cc)
        r = NL(cc)
        training_features.append(mdp.solve_MDP(gamma=gamma, tol=tol, w = r,flag='init',init_state=0,length=traj_len).M)

    expert_values = np.zeros(len(Conts))
    agent_values = np.zeros(len(Conts))
    contexts_seen = np.zeros(len(Conts))
    context_count = 0

    accuracy = []
    weights = [np.load("../../data/autonomous_driving/online/init_W_" + str(trainset) + ".npy")]
    weights = [mat/np.sum(mat.flatten()) for mat in weights]

    weights_mean = [mat.copy() for mat in weights]

    for t in range(iters):

        # Calculate values on test set:
        if (RUN_TEST and t % 5 == 0):
            test_agent_value_ar = []
            test_agent_loss_ar = []
            accur = np.zeros(test_size)

            for jk in range(test_size):
                agent_feat_exp_zz = feat_exp(NL_est(testset[jk],weights_mean),policy=True)
                test_agent_loss_ar.append(abs(((1-gamma)/3) * testset[jk] @ weights_mean[0] @ (agent_feat_exp_zz.M - expert_feat_exp_z[jk])))
                test_agent_value_ar.append(((1-gamma)/3) * testset[jk] @ (real_W) @ (agent_feat_exp_zz.M))
                accur[jk] = accuracy_mesure(agent_feat_exp_zz.policy,policies_expert[jk])

                if test_agent_loss_ar[-1] < 0:
                    test_agent_loss_ar[-1] *= (-1)
            test_agent_value = np.asarray(test_agent_value_ar).mean()
            test_agent_loss = np.asarray(test_agent_loss_ar).mean()
            accuracy.append(accur.mean())

            d[trainset,"test_value",t] = test_agent_value
            d[trainset,"test_loss",t] = test_agent_loss
            print(" == Generalization for ",str(t), "iterations: \n Value: " + str(test_agent_value) + "\nLoss: " + str(test_agent_loss))
            print("Accuracy: ", accur.mean())

        rand_batch = np.asarray(random.choices(list(range(len(training_contexts))), k=batch_size))
        weights = Update_estimator(weights, np.asarray(training_contexts)[t*batch_size:(t+1)*batch_size],
                                       np.asarray(training_features)[t*batch_size:(t+1)*batch_size], max_iter=1,
                                       stepsize=0.1, decay=1.0/np.sqrt(2*3*3*(t+1)), std=0.001,
                                       num_pts=500, batch_size=batch_size,
                                       epsilon=1e-3)

        weights_mean = [(weights_mean[0] * t + weights[0]) / (t + 1)]

    # save data:
    np.save("obj/accuracy_es_" + str(trainset) + ".npy",np.asarray(accuracy))
    d[trainset,"expert_values"] = expert_values
    d[trainset,"agent_values"] = agent_values
    d[trainset,"contexts_seen"] = contexts_seen

save_obj(d, "es_values"+str(valuesindex))
