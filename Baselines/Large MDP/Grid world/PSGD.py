#######################################################################################################################
#   file name: PSGD
#
#   description:
#   this file builds a grid-world environment.
#   then runs PSGD to solve the linear COIRL problem.
#   evaluates on a "large MDP"
#######################################################################################################################
# imports:
#######################################################################################################################
import sys
sys.path.append('../../../include')
import numpy as np
from ICMDP import *
from domains import *
import os
import time
import random
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
# HYPER PARAMS
#######################################################################################################################
gamma = 0.9
repeats = 5
tol = 1e-4
cont_nums = [5,10,15,20,30,40]
iteration_nums = np.arange(1,60)
batch_size = 6
dims = [3,4]
RUN_TEST = True

#######################################################################################################################
# Train & Evaluation
#######################################################################################################################
d = {}
dim_sq = dims[0]*dims[1]
d["iteration_nums"] = np.asarray(iteration_nums)
for trainset in range(repeats):
    for num_contexts in cont_nums:
        contexts = np.load("../../../data/grid_world/rand_contexts_" + str(trainset) + ".npy")[:num_contexts]
        mdp = build_domain('GRID', [dims[0], dims[1], False])
        training_features = []
        W_sol = []
        runtimes = []
        real_W = np.eye(dim_sq)
        for ctt in contexts:
            training_features.append(mdp.solve_MDP(gamma,tol=tol,w=(ctt @ real_W),flag='uniform').M)
        num_checks = 5
        for max_iters in iteration_nums:
            print(max_iters)
            runtime = 0.0
            for checki in range(num_checks):

                W = np.ones([dim_sq, dim_sq])
                W /= np.sum(W.flatten())
                W_mean = W.copy()
                # record time:
                start_time = time.time()
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

                    W = W - 0.1/ np.sqrt(2 * dim_sq * dim_sq * (iter + 1)) * batch_outer
                    W /= max((np.linalg.norm(np.reshape(W, W.size), np.inf)), 1.0)
                    W_mean = (W_mean * iter + W) / (iter + 1)

                end_time = time.time()
                W_sol.append(W_mean)
                runtime += end_time - start_time

            runtime /= float(num_checks)
            runtimes.append(runtime)
        d[trainset, "runtime", num_contexts] = runtimes
        mdp = build_domain('GRID', [dims[0],dims[1],True,contexts])

        def feat_exp(r):
            return mdp.solve_MDP(gamma=gamma,tol=tol,w=r,flag = 'uniform')

        real_W = (np.eye(dim_sq)).flatten()
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
        # Evaluate expert on test set
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

            if trainset > 0:
                save_obj(d, "psgd_values"+str(valuesindex))
            agent_vals = []
            agent_accurs = []
            for max_iters in iteration_nums:
                test_agent_value = 0.0
                test_agent_accuracy = 0.0
                for ncheck in range(num_checks):
                    solution = mdp.solve_MDP(gamma,tol=tol,w=W_sol[num_checks*(max_iters-1) + ncheck].flatten(),flag='uniform')

                    test_agent_value += np.asarray(real_W @ solution.M)
                    test_agent_accuracy += 100.0 * np.asarray(np.sum(policy_expert == solution.policy) / float(len(policy_expert))).mean()

                test_agent_accuracy /= float(num_checks)
                agent_accurs.append(test_agent_accuracy)
                test_agent_value /= float(num_checks)
                agent_vals.append(test_agent_value)
                print(" == Generalization for ",str(num_contexts), "contexts and ", str(max_iters)," iterations: " ,str(test_agent_value), ", runtime: ",str(runtimes[max_iters - 1]), ", accuracy: ",str(test_agent_accuracy))
            d[trainset,"test_value",num_contexts] = agent_vals
            d[trainset,"accuracy",num_contexts] = agent_accurs

save_obj(d, "psgd_values"+str(valuesindex))

