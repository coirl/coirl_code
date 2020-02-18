#######################################################################################################################
#   file name: ES
#
#   description:
#   this file builds a driving simulator environment.
#   then runs ES method to solve the linear COIRL problem.
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

# load real W:
real_W = np.load("../../../data/autonomous_driving/offline/realW.npy")

#######################################################################################################################
# Helper functions
#######################################################################################################################
def feat_exp(r):
    return mdp.solve_MDP(gamma=gamma,tol=tol,w=r,flag = 'init').M

def NL(x):
    return x @ real_W

def NL_est(x,mat_list):
    y = x
    for i in range(len(mat_list)):
        y = y @ mat_list[i]
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
            # Evaluate function and update step
            eval_plus = evaluate(step_plus, training_contexts, expert_feat_exp, agent_feat_exp)
            eval_minus = evaluate(step_minus, training_contexts, expert_feat_exp, agent_feat_exp)
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
d = {}
    
test_expert_value = 0

# Evaluate expert on test set
if RUN_TEST:
    test_expert_value = np.asarray([((1-gamma)/3) * NL(c) @ feat_exp(NL(c)) for c in testset]).mean()
    d["test_value"] = test_expert_value
    print("Expert test value: ",str(test_expert_value))

# run seeds:
for trainset in range(repeats):
    if trainset>0:
        save_obj(d, "es_values"+str(valuesindex))

    # load train set of contexts:
    Conts = np.load("../../../data/autonomous_driving/all/train_set_" + str(trainset) + ".npy")

    expert_values = np.zeros(len(Conts))
    agent_values = np.zeros(len(Conts))
    cum_regret = np.zeros(len(Conts))
    contexts_seen = np.zeros(len(Conts))
    context_count = 0

    weights = []
    weights.append(np.random.normal(size=(3,3)).reshape(3,3))
    weights = [mat/np.linalg.norm(mat.flatten()) for mat in weights]
    training_contexts = []
    training_features = []

    for t in range(iters):
        print("test ",trainset," timestep ",t, " Contexts seen:",len(training_contexts))

        # Agent and teacher play:
        ct = Conts[t]
        r = NL(ct)
        r_est = NL_est(ct,weights)
        features_expert = feat_exp(r)
        features_agent = feat_exp(r_est)
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
            test_agent_value = np.asarray([((1-gamma)/3) * NL(c) @ feat_exp(NL_est(c,weights)) for c in testset]).mean()
            d[trainset,"test_value",contexts_seen[t]] = test_agent_value
            print(" == Generalization for ",str(contexts_seen[t]), "contexts: ",str(test_agent_value))

        # If agent is more than epsilon suboptimal, update W:
        print("Value expert: ",str(value_expert)," Value agent: ",str(value_agent))
        if value_expert - value_agent > epsilon:
            training_contexts.append(ct)
            training_features.append(features_expert)
            print("Training with ",str(len(training_contexts)), " contexts:")
            weights = Update_estimator(weights, np.asarray(training_contexts),
                                       np.asarray(training_features), max_iter=50,
                                       stepsize=0.1, decay=0.95, std=0.001,
                                       num_pts=250, batch_size=len(training_contexts),
                                       epsilon=epsilon)

            context_count += 1

    # save data:
    d[trainset,"expert_values"] = expert_values
    d[trainset,"agent_values"] = agent_values
    d[trainset,"contexts_seen"] = contexts_seen
    d[trainset,"cum_regret"] = cum_regret

save_obj(d, "es_values"+str(valuesindex))
