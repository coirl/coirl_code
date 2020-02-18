#######################################################################################################################
#   file name: AL
#
#   description:
#   this file builds a grid-world environment.
#   then runs Apprenticeship Learning algorithm (Abbeel & Ng, 2004) to solve the COIRL problem via "large MDP"
#######################################################################################################################
# imports:
#######################################################################################################################
import sys
sys.path.append('../../../include')
import numpy as np
from ICMDP import *
from domains import *
import os
import pickle
import time

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
repeats = 5
tol = 1e-4
cont_nums = [5,10,15,20,30,40]
iteration_nums = np.arange(1,100)
dims = [3,4]
RUN_TEST = True

#######################################################################################################################
# Train & Evaluation
#######################################################################################################################
d = {}
d["iteration_nums"] = np.asarray(iteration_nums)
for trainset in range(repeats):
    for num_contexts in cont_nums:
        contexts = np.load("../../../data/grid_world/rand_contexts_" + str(trainset) + ".npy")[:num_contexts]

        mdp = build_domain('GRID', [dims[0],dims[1],True,contexts])

        def feat_exp(r):
            return mdp.solve_MDP(gamma=gamma,tol=tol,w=r,flag = 'uniform')

        real_W = (np.eye(dims[0]*dims[1])).flatten()
        real_W /= np.sum(real_W)

        def evaluate_verbose(model, training_contexts_all, expert_feat_exp_all):
            agent_r_est = [model.predict(np.expand_dims(ctest,axis=0))[0] for ctest in training_contexts_all]
            agent_feat_exp = [feat_exp.M(model.predict(np.expand_dims(ctest,axis=0))[0]) for ctest in training_contexts_all]
            value_err = np.asarray([(agent_r_est[j] @ (agent_feat_exp[j] - expert_feat_exp_all[j]))**2 \
                                    for j in range(len(training_contexts_all))]).sum()
            feat_err  = np.asarray([np.linalg.norm(agent_feat_exp[j] - expert_feat_exp_all[j],2) \
                                    for j in range(len(training_contexts_all))]).sum()
            dir_err = np.asarray([(agent_r_est[j] @ (agent_feat_exp[j] - expert_feat_exp_all[j])) \
                                    for j in range(len(training_contexts_all))]).sum()
            print("value err: ",value_err, " feat_exp err: ",feat_err, " dir err: ",dir_err)


        test_expert_value = 0
        expert_feat_exp_z = []
        test_expert_value_ar = []
        if RUN_TEST:
            feat_expert_i = feat_exp(real_W)
            policy_expert = feat_expert_i.policy
            feat_expert = feat_expert_i.M
            expert_feat_exp_z.append(feat_expert)
            test_expert_value_ar.append(real_W @ feat_expert)
            test_expert_value = np.asarray(test_expert_value_ar).mean()
            d[trainset,"test_value_expert",num_contexts] = test_expert_value
            print("Expert test value: ",str(test_expert_value))


            num_checks = 5
            if trainset > 0:
                save_obj(d, "al_values"+str(valuesindex))
            agent_vals = []
            runtimes = []
            agent_accurs = []
            for iterz in iteration_nums:
                # record time:
                runtime = 0.0
                test_agent_value = 0.0
                test_agent_accuracy = 0.0
                for ncheck in range(num_checks):
                    start_time = time.time()
                    solution = mdp.solve_IMDP_PROJ(gamma, iterz, feat_expert, tol=tol, flag='uniform')
                    end_time = time.time()
                    test_agent_value += real_W @ solution.M[-1,:]
                    runtime += end_time - start_time

                    test_agent_accuracy += 100.0 * np.asarray([pol[0]*np.sum(policy_expert == pol[1:]) / float(len(policy_expert)) for pol in solution.policies]).sum()
               
                test_agent_accuracy /= float(num_checks)
                agent_accurs.append(test_agent_accuracy)
                runtime /= float(num_checks)
                test_agent_value /= float(num_checks)
                agent_vals.append(test_agent_value)
                runtimes.append(runtime)
                print(" == Generalization for ",str(num_contexts), "contexts and ",str(iterz), "iterations: ",str(test_agent_value), ", runtime: ",str(runtime), ", accuracy: ",str(test_agent_accuracy))
            d[trainset,"test_value",num_contexts] = agent_vals
            d[trainset,"runtime",num_contexts] = runtimes
            d[trainset,"accuracy",num_contexts] = agent_accurs


save_obj(d, "al_values"+str(valuesindex))
