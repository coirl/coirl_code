import numpy as np
import scipy.io as sio
from ICMDP import *
import os
from sklearn.cluster import KMeans
import random
import pickle
from scipy import stats
from sklearn.mixture import GaussianMixture
from sklearn import metrics
from sklearn.cluster import KMeans
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Dropout, BatchNormalization, LSTM, Embedding, Concatenate, CuDNNLSTM, RepeatVector, Softmax
from keras.layers import LeakyReLU, Bidirectional, MaxPooling2D, Conv2D, Flatten, InputLayer, AveragePooling2D, Lambda, ReLU
from keras.layers.merge import add
from keras import backend as K
from keras import regularizers
from keras import optimizers
from scipy.special import softmax
from sklearn.preprocessing import PolynomialFeatures

import sys
valuesindex = 0
testi = 0
testf = 5
def save_obj(obj, name):
    with open('obj/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)
#########################################################################################
sepsis_data = np.genfromtxt('MIMICtable_e.csv', delimiter=',')
# 0-2: time, place within trajectory, patient id
# 3-9: context
# 10 - patient died within hospital, ignore for now
# 11-12: actions
# 13- : features
# build trajectories:
features = []
trajectory_features = []
actions = []
trajectory_actions = []
contexts = []
contextsr = []
trajectory_length = []
died_flag = []
died_flagr = []
ww = [0, 0, 0, 0, 0, 0,1]
weight_list = [ww[6],ww[4], ww[6], ww[6], ww[4], ww[4], ww[3], ww[3], ww[4], ww[3], ww[3], ww[3], ww[3], ww[4], ww[4], ww[3],
               ww[3], ww[4], ww[4], ww[3], ww[3], ww[3], ww[3], ww[4], ww[3], ww[4], ww[3], ww[4], ww[4], ww[5], ww[3],
               ww[3], ww[4], ww[5], ww[5], ww[6], ww[4], ww[4], ww[6], ww[4], ww[5], ww[4],ww[4]]
weight_list = np.asarray(weight_list)
cont_w_list = np.asarray([0,1.0,0,1.0,1.0,0,2.0])
cont_w_list = np.ones(cont_w_list.shape)
# extract data:
gamma=0.7
prev_len = 0
skip_flag = False
last_flag = False
for i in range(sepsis_data.shape[0]):
    # new trajectory:
    if sepsis_data[i][0] == 1 or sepsis_data[i][0] < sepsis_data[i - 1][0]:
        if sepsis_data[i][7] <= 10:
            skip_flag = True
            continue
        else:
            skip_flag = False
            trajectory_features.append([])
            trajectory_actions.append([])
            trajectory_length.append(0)
            contexts.append(sepsis_data[i][3:10][cont_w_list > 0])
            died_flag.append(sepsis_data[i][10])
            if trajectory_length[0] != 0 and trajectory_length[-2] > 10:
                last_flag = False
                features.extend(trajectory_features[-2])
                contextsr.append(contexts[-2])
                died_flagr.append(died_flag[-2])
        last_flag = True
    if skip_flag:
        continue
    else:
        trajectory_length[-1] += 1
        # features.append(sepsis_data[i][13:])
        trajectory_features[-1].append(sepsis_data[i][12:][weight_list > 0])
        actions.append(sepsis_data[i][11:13])
        actions[-1][1] = (1.0 / contexts[-1][4])*sepsis_data[i][12]
        trajectory_actions[-1].append(sepsis_data[i][11:13])
        trajectory_actions[-1][-1][1] = (1.0 / contexts[-1][4])*sepsis_data[i][12]
ccc = 0
for cd in range(len(trajectory_features)):
    if trajectory_length[cd] > 10:
        ccc += len(trajectory_features[cd])
if last_flag and trajectory_length[-1] > 10:
    features.extend(trajectory_features[-1])
    contextsr.append(contexts[-1])
    died_flagr.append(died_flag[-1])
Contexts = np.asarray(contexts)
ContextsR = np.asarray(contextsr)
ContextsR = (ContextsR - np.mean(ContextsR, axis=0)) / np.std(ContextsR, axis=0)

Actions = np.asarray(actions)
vaso = Actions[:, 0].copy()
fluid = Actions[:, 1].copy()

# create actions:
ind_vas = (vaso != 0)
ind_fluid = (fluid != 0)
ranked_vaso = stats.rankdata(vaso[ind_vas]) / vaso[ind_vas].size
ranked_fluid = stats.rankdata(fluid[ind_fluid]) / fluid[ind_fluid].size

bins1 = 5
bins2 = 5
nActions = bins1*bins2

ranked_vaso = np.floor((bins1 - 1) * (ranked_vaso + (1.0 /(bins1 - 1)) - 0.0000000001))
if bins2 == 1:
    ranked_fluid = np.zeros(ranked_fluid.shape)
else:
    ranked_fluid = np.floor((bins2 - 1) * (ranked_fluid +  (1.0 /(bins2 - 1)) - 0.0000000001))

vaso[ind_vas] = ranked_vaso
fluid[ind_fluid] = ranked_fluid
curr_ind = 0
for i in range(len(trajectory_actions)):
    for j in range(len(trajectory_actions[i])):
        act_vaso = vaso[curr_ind]
        act_fluid = fluid[curr_ind]
        curr_ind += 1
        trajectory_actions[i][j] = int(bins2 * act_vaso + act_fluid)

trajectory_length = np.asarray(trajectory_length)
trajectory_actions = np.asarray(trajectory_actions)
trajectory_actions = trajectory_actions[trajectory_length > 10]

print(len(trajectory_length))
trajectory_length = trajectory_length[trajectory_length > 10]
# compare models:
repeats = 5
trajectory_features_i = trajectory_features.copy()
test_size = int(0.2*len(trajectory_length))
valid_size = int(0.2*len(trajectory_length))

nums_clusters = [1000]
weight_list = weight_list[weight_list > 0]
KM_features = np.asarray(features)
KM_features = (KM_features - np.mean(KM_features, axis=0)) / np.std(KM_features, axis=0)
for jj in range(len(weight_list)):
    KM_features[:, jj] = weight_list[jj] * KM_features[:, jj]
print(KM_features.shape)
Kmclusters = np.load("KM_clusters.npy")
Kmlabels = np.load("KM_labels.npy")
for trainset in range(1):
    for nclusters in nums_clusters:
        d={}
        for trainset in range(testi,testf):
            if trainset > 0:
                save_obj(d, "coirl_values"+str(valuesindex))
            transitions_all = []
            end_states_all = []
            died_flagr = np.asarray(died_flagr)
            rand_inds = np.arange(trajectory_length.shape[0])
            random.shuffle(rand_inds)
            test_inds = rand_inds[:test_size]
            valid_inds = rand_inds[test_size:test_size+valid_size]
            train_inds = rand_inds[test_size+valid_size:]

            # contextual dynamics
            n_trans = 1
            cont_w_list = cont_w_list[cont_w_list > 0]
            kmeans_contexts = KMeans(n_clusters=n_trans, random_state=0).fit(ContextsR[:,[1,4]])
            Kmlabels_contexts = kmeans_contexts.labels_
            for trans_ii in range(n_trans):
                end_states_all.append(np.zeros(nclusters))
                transitions = []
                for i in range(nclusters + 1):
                    transitions.append([])
                    for j in range(nActions):
                        transitions[i].append([])
                        for k in range(nclusters + 1):
                            transitions[i][j].append(0)
                transitions_all.append(transitions)
    
            expert_traj = []
            expert_init_states = []
            tot_traj = 0
            for i in range(trajectory_length.size):
                trajectory = []
                for j in range(trajectory_length[i]):
                    trajectory.append(Kmlabels[tot_traj+j])
                tot_traj += int(trajectory_length[i])
                expert_init_states.append(trajectory[0])
                expert_traj.append(np.asarray(trajectory))
    
                if i in train_inds:
                    trans_no = Kmlabels_contexts[i]
                    if died_flagr[i] ==  0:
                        end_states_all[trans_no][trajectory[-1]] += 1
                    for k in range(int(trajectory_length[i])-1):
                        transitions_all[trans_no][trajectory[k]][trajectory_actions[i][k]-1][trajectory[k+1]] += 1
            
            e_states = []
            for ttrr in range(n_trans):
                e_states.append( np.arange(nclusters)[end_states_all[ttrr] > 20])

            THETA = []
            for trans_ii in range(n_trans):
                THETA.append(Transitions(num_states=nclusters + 1, num_actions=nActions))
                for i in range(nclusters):
                    for j in range(nActions):
                        sum_trans = 0
                        for k in range(nclusters):
                            sum_trans += transitions_all[trans_ii][i][j][k]
                        if (sum_trans == 0):
                            THETA[-1].set_trans(i,j,nclusters,1)
                        else:
                            for k in range(nclusters):
                                prob = float(transitions_all[trans_ii][i][j][k])/float(sum_trans)
                                THETA[-1].set_trans(i,j,k,prob)
                for i in range(nActions):
                    THETA[-1].set_trans(nclusters,i,nclusters,1)

            mdp = ICMDP()
            mdp.set_THETA(THETA[0])
            for trans_ii in range(n_trans):
                THETA[trans_ii] = THETA[trans_ii].mat_form()
    
    
            dim_features = Kmclusters.shape[1] + 1
            F = Features(dim_features=dim_features)
            feature = np.zeros(dim_features)
    
            for i in range(nclusters):
                feature[:-1] = Kmclusters[i,:]
                feature[-1] = 1.0
                F.add_feature(feature=feature)
    
            feature = np.zeros(dim_features)
            feature[-1] = -1.0
            F.add_feature(feature=feature)
    
            mdp.set_F(F)
            lr_init = 1.0
    
            def feat_exp(r,init_state,FH=False):
                return mdp.solve_MDP(gamma=gamma,tol=tol,w=r,flag = 'init',init_state=init_state,action_features=True,FH=FH)
    
            def custom_loss():
                def loss(y_true, y_pred):
                    return K.batch_dot(y_pred, y_true, axes=1)
    
                return loss
    
    
            def reset_weights(model):
                session = K.get_session()
                for layer in model.layers:
                    if hasattr(layer, 'kernel_initializer'):
                        layer.kernel.initializer.run(session=session)
            print(ContextsR[0].size)    
            exit()
            def buildmodel():
                l = 0.001
                input1 = Input(shape=(ContextsR[0].size,))
                x = Dense(20* dim_features, use_bias=False)(input1)
                x = BatchNormalization()(x)
                x = LeakyReLU(alpha=0.1)(x)
                x = Dense(10 * dim_features, use_bias=False)(x)
                x = LeakyReLU(alpha=0.1)(x)

                x = Dense(int(contextsr[0].size/2) * dim_features, use_bias=True)(x)

                x = Dense(dim_features + nActions, use_bias=False)(x)
                output1 = Lambda(lambda a: K.l2_normalize(a + 1e-10, axis=1))(x)
                model = Model(inputs=[input1], outputs=[output1])
                myopt = optimizers.SGD(lr=0.1, decay=1e-7, momentum=0.9, nesterov=True)
                model.compile(loss=custom_loss(), optimizer=myopt)
                return model
    
    
            def Update_estimator(model,  contexts_dynamics, training_contexts_all, init_states_all, expert_feat_exp_all, max_iter=120, batch_size=1,
                                 debug_interval=50, epsilon=1e-3, lr_init=0.1, decay=0.95,traj_len=False):
                for iteration in range(1, max_iter + 1):
                    # calculate feature expectations for current parameters, for all contexts in train set
                    agent_feat_exp_all = model.predict(training_contexts_all)
    
                    # select batch
                    indices = np.random.choice(a=list(range(len(training_contexts_all))), size=batch_size, replace=False)
                    training_contexts = training_contexts_all[indices]
                    expert_feat_exp = expert_feat_exp_all[indices]
                    init_states = init_states_all[indices]
                    agent_feat_exp = agent_feat_exp_all[indices]
                    contexts_dynamics_i = contexts_dynamics[indices]
                    if traj_len:
                        traj_len_i = traj_len[indices]
    
                    for aa in range(len(agent_feat_exp)):
                        mdp.set_THETA_mat(THETA[contexts_dynamics_i[aa]])
                        agent_feat_exp[aa] = np.array(feat_exp(agent_feat_exp[aa], init_states[aa]).M)
    
                    feat_exp_diff = agent_feat_exp - expert_feat_exp
    
                    # Take step
                    LEARNING_RATE = lr_init * (decay ** iteration)
                    K.set_value(model.optimizer.lr, LEARNING_RATE)
                    model.fit(training_contexts, feat_exp_diff, batch_size=batch_size, epochs=1, verbose=0)
    
                return model
        #########################################################################################
    
            #HYPER PARAMS
            batch_size = 32
            iters = 50
            tol = 1e-3
            RUN_TEST = True
            n_static = ContextsR.shape[1]
            lr_init = 1.0
    

            m = buildmodel()
            died_flagr = np.asarray(died_flagr)
            died_test = died_flagr[test_inds]
            died_valid = died_flagr[valid_inds]
            died_train = died_flagr[train_inds]
            testset = ContextsR[test_inds]
            validset = ContextsR[valid_inds]
            test_cont_dynamics = Kmlabels_contexts[test_inds]
            valid_cont_dynamics = Kmlabels_contexts[valid_inds]
            test_init_states = np.asarray(expert_init_states)[test_inds]
            valid_init_states = np.asarray(expert_init_states)[valid_inds]
            train_contexts = ContextsR[train_inds]
            train_cont_dynamics = Kmlabels_contexts[train_inds]
            train_init_states = np.asarray(expert_init_states)[train_inds]
            train_trajectory_states = np.asarray(expert_traj)[train_inds]
            test_trajectory_states = np.asarray(expert_traj)[test_inds]
            valid_trajectory_states = np.asarray(expert_traj)[valid_inds]
            train_trajectory_actions = trajectory_actions[train_inds]
            test_trajectory_actions = trajectory_actions[test_inds]
            valid_trajectory_actions = trajectory_actions[valid_inds]
    
    
            training_contexts = []
            training_featuresss = []
            training_inits = []
            train_expert_valuess = []
            train_expert_policy = []
            for ind in range(len(train_init_states)):
                training_contexts.append(train_contexts[ind])
                training_inits.append(train_init_states[ind])
                trajectory_feat_exp = np.zeros(dim_features+nActions)
                curr_g = 1.0
                curr_pol = -1 * np.ones(nclusters + 1,dtype=int)
                for i in range(trajectory_length[train_inds[ind]]):
                    ext_feat = np.zeros(nActions+1)
                    ext_feat[train_trajectory_actions[ind][i] + 1] = 1.0
                    curr_st = train_trajectory_states[ind][i]
                    trajectory_feat_exp += curr_g * np.append(Kmclusters[curr_st],ext_feat)
                    curr_g *= gamma
                    curr_pol[curr_st] = train_trajectory_actions[ind][i]
                training_featuresss.append(trajectory_feat_exp)
                train_expert_policy.append(curr_pol)
    
            if RUN_TEST:
                policies_expert = []
                features_expert_z = []
                for jk in range(len(test_init_states)):
                    trajectory_feat_exp = np.zeros(dim_features)
                    curr_g = 1.0
                    curr_pol = -1 * np.ones(nclusters + 1, dtype=int)
                    for i in range(trajectory_length[test_inds[jk]]):
                        curr_st = test_trajectory_states[jk][i]
                        trajectory_feat_exp[:-1] += curr_g * Kmclusters[curr_st]
                        curr_g *= gamma
                        curr_pol[curr_st] = test_trajectory_actions[jk][i]
                    policies_expert.append(curr_pol)
                    features_expert_z.append(trajectory_feat_exp)
                
                policies_expert_valid = []
                features_expert_z_valid = []
                for jk in range(len(valid_init_states)):
                    trajectory_feat_exp = np.zeros(dim_features)
                    curr_g = 1.0
                    curr_pol = -1 * np.ones(nclusters + 1, dtype=int)
                    for i in range(trajectory_length[valid_inds[jk]]):
                        curr_st = valid_trajectory_states[jk][i]
                        trajectory_feat_exp[:-1] += curr_g * Kmclusters[curr_st]
                        curr_g *= gamma
                        curr_pol[curr_st] = valid_trajectory_actions[jk][i]
                    policies_expert_valid.append(curr_pol)
                    features_expert_z_valid.append(trajectory_feat_exp)

            train_rand_inds = np.arange(len(train_init_states))
            W_init = 2 * np.random.rand(training_contexts[0].size,dim_features) - 1
            W_init /= np.linalg.norm(W_init.flatten(), 2)
            random.shuffle(train_rand_inds)
            Conts = train_contexts[train_rand_inds].copy()
            Tlen = trajectory_length[train_inds][train_rand_inds]
            testTlen = trajectory_length[test_inds]
            validTlen = trajectory_length[valid_inds]
            Conts_died = died_train[train_rand_inds].copy()
        #
            Conts_dyn = train_cont_dynamics[train_rand_inds].copy()
            Train_features = np.asarray(training_featuresss)[train_rand_inds].copy()
            Train_policies = np.asarray(train_expert_policy)[train_rand_inds].copy()
            train_inits = np.asarray(train_init_states)[train_rand_inds].copy()

            accuracy = []
            action_dist = []

            W = W_init.copy()
            W_mean = W.copy()

            num_contexts = len(train_inds)
            reset_weights(m)
            ttlen = Tlen[:num_contexts]
            contexts = Conts[:num_contexts]
            training_inits = train_inits[:num_contexts]
            contexts_dynamics = Conts_dyn[:num_contexts]
            training_features = Train_features[:num_contexts]
            training_policy = Train_policies[:num_contexts]
            W_sol = []
            d[trainset,"accuracy_d2",num_contexts] = []
            d[trainset,"accuracy_d2_valid",num_contexts] = []
            donee = False

            best_accur = 0
            for iter in range(iters):
               if donee: 
                    agent_accurs = []
                    test_agent_accur_ar = []

                    for cont in range(len(testset)):
                        r_est = m.predict(np.expand_dims(testset[cont], axis=0))[0]
                        mdp.set_THETA_mat(THETA[test_cont_dynamics[cont]])
                        solution = mdp.solve_MDP(gamma,tol=tol,w=r_est,flag='init',init_state=test_init_states[cont],top_policy=5,action_features=True)
                        policy_expert_i = policies_expert[cont]
                        test_agent_accur = 0.0
                        for pp in np.arange(bins2):
                            test_agent_accur += 100 * np.asarray(np.sum((policy_expert_i[policy_expert_i >= 0] == ((solution.policy[:,0][policy_expert_i >= 0] - (solution.policy[:,0][policy_expert_i >= 0] % bins2)) + pp)) / float(len(policy_expert_i[policy_expert_i >= 0])))).mean()

                        test_agent_accur_ar.append(test_agent_accur)


                    test_agent_accuracy = np.asarray(test_agent_accur_ar).mean()
                    agent_accurs.append(test_agent_accuracy)

                    print("===================== Evaluation for ", str(iter), " iterations:")
                    print(" == Generalization for ",str(num_contexts), "contexts: ", ", accuracy: ",str(test_agent_accuracy))
                    d[trainset,"accuracy",num_contexts].append(agent_accurs)
                    break

               rand_sample = random.choices(list(range(num_contexts)), k=batch_size)
               m = Update_estimator(model=m,contexts_dynamics=contexts_dynamics,training_contexts_all=np.asarray(contexts[rand_sample]),
                                    init_states_all=np.asarray(training_inits[rand_sample]),
                                    expert_feat_exp_all=np.array(training_features[rand_sample]),
                                    max_iter=1, batch_size=batch_size,
                                    debug_interval=99999, epsilon=5e-4, lr_init=0.2,
                                    decay=0.92 ** iter)

               valid_agent_accurs = []
               valid_agent_accur_ar = []
               for cont in range(200):
                   r_est = m.predict(np.expand_dims(validset[cont], axis=0))[0]
                   mdp.set_THETA_mat(THETA[valid_cont_dynamics[cont]])
                   solution = mdp.solve_MDP(gamma,tol=tol,w=r_est,flag='init',init_state=valid_init_states[cont],top_policy=5,action_features=True)
                   policy_expert_i = policies_expert[cont]
                   valid_agent_accur = 0.0
                   for pp in np.arange(bins2):
                       valid_agent_accur += 100 * np.asarray(np.sum((policy_expert_i[policy_expert_i >= 0] == ((solution.policy[:,0][policy_expert_i >= 0] - (solution.policy[:,0][policy_expert_i >= 0] % bins2)) + pp)) / float(len(policy_expert_i[policy_expert_i >= 0])))).mean()

                   valid_agent_accur_ar.append(valid_agent_accur)

               valid_agent_accuracy = np.asarray(valid_agent_accur_ar).mean()
               valid_agent_accurs.append(valid_agent_accuracy)

               print("===================== Validation for ", str(iter), " iterations:")
               print(" == Validation for ",str(num_contexts), "contexts: ", ", accuracy: ",str(valid_agent_accuracy))
               d[trainset,"accuracy_valid",num_contexts].append(valid_agent_accurs)
               if (best_accur > 3 + valid_agent_accuracy) and (iter >= 5):
                   donee = True
                   m.set_weights(best_weights)
               if best_accur < valid_agent_accuracy:
                  best_accur = valid_agent_accuracy
                  best_weights = m.get_weights()


            save_obj(d, "coirl_values"+str(valuesindex))

        save_obj(d, "coirl_values"+str(valuesindex))
