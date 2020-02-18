#######################################################################################################################
#   file name: EW
#
#   description:
#   this file builds a driving simulator environment.
#   then runs EW to solve the linear COIRL problem.
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
from accuracy import *

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
traj_len = 40
batch_size = 1
iters = 500
train_size = 1000
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
# define functions for EW
#######################################################################################################################

def feat_exp(r, policy = False):
    if policy:
        return mdp.solve_MDP(gamma=0.9,tol=tol,w=r,flag = 'init')
    return mdp.solve_MDP(gamma=0.9,tol=tol,w=r,flag = 'init').M

def get_random_W(n):
    x = np.random.exponential(scale=1.0, size=n)
    y = [x[0:i].sum()/x.sum() for i in range(len(x)+1)]
    z = [y[i]-y[i-1] for i in range(1,len(y))]
    return np.asarray(z)

def NL(x):
    return x @ real_W/np.linalg.norm(real_W.flatten(),2)

def evaluate_verbose(model, training_contexts_all, expert_feat_exp_all):
    agent_r_est = [model.predict(np.expand_dims(ctest,axis=0))[0] for ctest in training_contexts_all]
    agent_feat_exp = [feat_exp(model.predict(np.expand_dims(ctest,axis=0))[0]) for ctest in training_contexts_all]
    value_err = np.asarray([(agent_r_est[j] @ (agent_feat_exp[j] - expert_feat_exp_all[j]))**2 \
                            for j in range(len(training_contexts_all))]).sum()
    feat_err  = np.asarray([np.linalg.norm(agent_feat_exp[j] - expert_feat_exp_all[j],2) \
                            for j in range(len(training_contexts_all))]).sum()
    dir_err = np.asarray([(agent_r_est[j] @ (agent_feat_exp[j] - expert_feat_exp_all[j])) \
                          for j in range(len(training_contexts_all))]).sum()
    print("value err: ",value_err, " feat_exp err: ",feat_err, " dir err: ",dir_err)


#######################################################################################################################
# Test the EW method in the environment
#######################################################################################################################
d = {}

test_expert_value = 0
expert_feat_exp_z = []
test_expert_value_ar = []
policies_expert = []
if RUN_TEST:
    for jk in range(test_size):
        feat_expert = feat_exp(NL(testset[jk]),policy=True)
        expert_feat_exp_z.append(feat_expert.M)
        policies_expert.append(feat_expert.policy)
        test_expert_value_ar.append(((1-0.9)/3) * (testset[jk] @ real_W) @ feat_expert.M)
    test_expert_value = np.asarray(test_expert_value_ar).mean()
    d["test_value"] = test_expert_value
    print("Expert test value: ",str(test_expert_value))


for trainset in range(repeats):
    if trainset>0:
        save_obj(d, "ew_values"+str(valuesindex))

    Conts = np.load("../../data/autonomous_driving/all/train_set_"+str(trainset)+".npy")[:train_size]
    training_contexts = []
    training_features = []
    for cc in Conts:
        training_contexts.append(cc)
        r = NL(cc)
        training_features.append(mdp.solve_MDP(gamma=0.9, tol=tol, w = r,flag='init',init_state=0,length=traj_len).M)

    accuracy = []
    expert_values = np.zeros(len(Conts))
    agent_values = np.zeros(len(Conts))
    contexts_seen = np.zeros(len(Conts))
    context_count = 0

    W = np.load("../../data/autonomous_driving/online/init_W_" + str(trainset) + ".npy")
    W_mean = W.copy()

    for t in range(iters):

        # Calculate values on test set
        if (RUN_TEST and t % 5 == 0):
            test_agent_value_ar = []
            test_agent_loss_ar = []
            accur = np.zeros(test_size)
            act_dist = np.zeros(test_size)
            for jk in range(test_size):
                agent_feat_exp_zz = feat_exp(testset[jk] @ W_mean/np.linalg.norm(W_mean.flatten(),2),policy=True)
                test_agent_loss_ar.append(abs(((1-0.9)/3) * (testset[jk] @ (W_mean)) @ (agent_feat_exp_zz.M - expert_feat_exp_z[jk])))
                accur[jk] = accuracy_mesure(agent_feat_exp_zz.policy,policies_expert[jk])

                test_agent_value_ar.append(((1-0.9)/3) * (testset[jk] @ real_W) @ agent_feat_exp_zz.M)
            test_agent_loss = np.asarray(test_agent_loss_ar).mean()
            test_agent_value = np.asarray(test_agent_value_ar).mean()
            accuracy.append(accur.mean())

            d[trainset,"test_value",t] = test_agent_value
            d[trainset,"test_loss",t] = test_agent_loss
            print(" == Generalization for ",str(t), "iterations: \n Value: " + str(test_agent_value) + "\nLoss: " + str(test_agent_loss))
            print("Accuracy: ", accur.mean())

        rand_sample = np.arange(t*batch_size,(t+1)*batch_size) # random.choices(list(range(len(training_contexts))),k=batch_size)
        r_est = training_contexts[rand_sample[0]] @ W/np.linalg.norm(W.flatten(),2)
        features_agent = feat_exp(r_est)
        batch_outer = np.outer(training_contexts[rand_sample[0]], features_agent-training_features[rand_sample[0]])
        for sample in rand_sample[1:]:
            r_est = training_contexts[sample] @ W/np.linalg.norm(W.flatten(),2)
            features_agent = feat_exp(r_est)
            batch_outer += np.outer(training_contexts[sample], features_agent-training_features[sample])

        batch_outer /= batch_size

        W = np.multiply(W,np.exp(-1*(1-0.9) *100* np.sqrt(np.log((real_W.shape[0])*real_W.shape[1])/(2*(t + 1))) * batch_outer))
        W /= np.sum(np.reshape(W,W.size))

        W_mean = (W_mean*t + W) / (t + 1)

    # save data:
    np.save("obj/accuracy_ew_" + str(trainset) + ".npy",np.asarray(accuracy))
    d[trainset,"expert_values"] = expert_values
    d[trainset,"agent_values"] = agent_values
    d[trainset,"contexts_seen"] = contexts_seen

save_obj(d, "ew_values"+str(valuesindex))
