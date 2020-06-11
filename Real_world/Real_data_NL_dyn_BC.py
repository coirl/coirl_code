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
from sklearn.decomposition import PCA
from sklearn.preprocessing import PolynomialFeatures
from keras.callbacks import EarlyStopping

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
##########################################################################################
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
tr_temp = []
trajectory_length = np.asarray(trajectory_length)
trajectory_actions = np.asarray(trajectory_actions)
trajectory_actions = trajectory_actions[trajectory_length > 10]
for ii in range(len(trajectory_length)):
    if trajectory_length[ii] > 10:
        tr_temp.append(trajectory_features[ii])
trajectory_features = tr_temp #trajectory_features[trajectory_length > 10]

trajectory_length = trajectory_length[trajectory_length > 10]

repeats = 5
trajectory_features_i = trajectory_features.copy()
# features = features[:nf]

###
test_size = int(0.2*len(trajectory_length))
valid_size = int(0.2*len(trajectory_length))

nums_clusters = [1000]
weight_list = weight_list[weight_list > 0]
KM_features = np.asarray(features)
KM_features = (KM_features - np.mean(KM_features, axis=0)) / np.std(KM_features, axis=0)
for jj in range(len(weight_list)):
    KM_features[:, jj] = weight_list[jj] * KM_features[:, jj]
Kmclusters = np.load("KM_clusters.npy")
Kmlabels = np.load("KM_labels_nclusters.npy")
d={}
#############################################################################
for trainset in range(5):

    rand_inds = np.arange(trajectory_length.shape[0]).astype(np.int)
    random.shuffle(rand_inds)
    test_inds = rand_inds[:test_size]
    valid_inds = rand_inds[test_size:test_size+valid_size]
    train_inds = rand_inds[test_size+valid_size:]

    transitions_all = []
    end_states_all = []
    died_flagr = np.asarray(died_flagr)
    n_trans = 5
    cont_w_list = cont_w_list[cont_w_list > 0]
    for cc in range(cont_w_list.size):
        ContextsR[:,cc] = cont_w_list[cc]*ContextsR[:,cc]
    kmeans_contexts = KMeans(n_clusters=n_trans, random_state=0).fit(ContextsR)
    kmlabels_contexts = kmeans_contexts.labels_
    for cc in range(cont_w_list.size):
        ContextsR[:,cc] = (1.0/cont_w_list[cc])*ContextsR[:,cc]
    for trans_ii in range(n_trans):
        end_states_all.append(np.zeros(nclusters))
        transitions = []
        for i in range(nclusters + 1):
            transitions.append([])
            for j in range(25):
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
            trans_no = kmlabels_contexts[i]
            if died_flagr[i] ==  0:
                end_states_all[trans_no][trajectory[-1]] += 1
            for k in range(int(trajectory_length[i])-1):
                transitions_all[trans_no][trajectory[k]][trajectory_actions[i][k]-1][trajectory[k+1]] += 1

    e_states = []
    for ttrr in range(n_trans):
        e_states.append( np.arange(nclusters)[end_states_all[ttrr] > 20])

    THETA = []
    for trans_ii in range(n_trans):
        THETA.append(Transitions(num_states=nclusters + 1, num_actions=25))
        for i in range(nclusters):
            for j in range(25):
                sum_trans = 0
                for k in range(nclusters):
                    sum_trans += transitions_all[trans_ii][i][j][k]
                if (sum_trans == 0):
                    THETA[-1].set_trans(i,j,nclusters,1)
                else:
                    for k in range(nclusters):
                        prob = float(transitions_all[trans_ii][i][j][k])/float(sum_trans)
                        THETA[-1].set_trans(i,j,k,prob)
        for i in range(25):
            THETA[-1].set_trans(nclusters,i,nclusters,1)

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

    lr_init = 1.0



    def reset_weights(model):
        session = K.get_session()
        for layer in model.layers:
            if hasattr(layer, 'kernel_initializer'):
                layer.kernel.initializer.run(session=session)

    print(Kmclusters.shape[1] + ContextsR.shape[1])
    exit()
    def buildmodel():
        input1 = Input(shape=(Kmclusters.shape[1] + ContextsR.shape[1],))
        x = Dense(625, use_bias=False)(input1)
        x = LeakyReLU(alpha=0.1)(x)
        x = Dense(125, use_bias=False)(x)
        x = Dense(25, use_bias=False)(x)
        output1 = Softmax(axis=-1)(x)
        model = Model(inputs=[input1], outputs=[output1])
        sgd = optimizers.SGD(lr=0.1, decay=1e-7, momentum=0.9, nesterov=True)
        model.compile(loss='binary_crossentropy', optimizer=sgd)
        return model


    #####################################################################################

    #HYPER PARAMS
    batch_size = 32
    iters = 1000
    tol = 1e-3
    RUN_TEST = True

    n_static = ContextsR.shape[1]
    nActions = 25


    lr_init = 1.0

    m = buildmodel()
    died_flagr = np.asarray(died_flagr)
    died_test = died_flagr[test_inds]
    died_train = died_flagr[train_inds]
    testset = ContextsR[test_inds]
    test_cont_dynamics = kmlabels_contexts[test_inds]
    test_init_states = np.asarray(expert_init_states)[test_inds]
    train_contexts = ContextsR[train_inds]
    train_cont_dynamics = kmlabels_contexts[train_inds]
    train_init_states = np.asarray(expert_init_states)[train_inds]
    train_trajectory_states = np.asarray(expert_traj)[train_inds]
    test_trajectory_states = np.asarray(expert_traj)[test_inds]
    train_trajectory_actions = trajectory_actions[train_inds]
    test_trajectory_actions = trajectory_actions[test_inds]

    valid_contexts = ContextsR[valid_inds]
    valid_cont_dynamics = kmlabels_contexts[valid_inds]
    valid_init_states = np.asarray(expert_init_states)[valid_inds]
    valid_trajectory_states = np.asarray(expert_traj)[valid_inds]
    valid_trajectory_actions = trajectory_actions[valid_inds]
    train_cont_dyn = kmlabels_contexts[train_inds]
    valid_cont_dyn = kmlabels_contexts[valid_inds]
    test_cont_dyn = kmlabels_contexts[test_inds]

    pca = PCA(n_components=Kmclusters.shape[1])
    Cpca = []
    for nt in range(n_trans):
        Cpca.append(pca.fit_transform(np.reshape(THETA[nt][:1000*25,:1000], [Kmclusters.shape[1], Kmclusters.shape[1] * 25])))
    for trainset1 in range(1):
        training_contexts = []
        training_featuresss = []
        training_inits = []
        train_expert_valuess = []
        train_expert_policy = []
        bc_inputs = []
        bc_outputs = []
        for ind in range(len(train_init_states)):
            training_contexts.append(train_contexts[ind])
            training_inits.append(train_init_states[ind])
            trajectory_feat_exp = np.zeros(dim_features+25)
            curr_g = 1.0
            curr_pol = -1 * np.ones(nclusters + 1,dtype=int)
            for i in range(trajectory_length[train_inds[ind]]):
                ext_feat = np.zeros(26)
                ext_feat[train_trajectory_actions[ind][i] + 1] = 1.0
                curr_st = train_trajectory_states[ind][i]
                trajectory_feat_exp += curr_g * np.append(Kmclusters[curr_st],ext_feat)
                curr_g *= gamma
                curr_pol[curr_st] = train_trajectory_actions[ind][i]
                bc_inputs.append(np.append(train_contexts[ind],Cpca[train_cont_dyn[ind]][curr_st]))
                bc_outputs.append(ext_feat[1:])

            training_featuresss.append(trajectory_feat_exp)
            train_expert_policy.append(curr_pol)

        bc_input_valid = []
        bc_output_valid = []
        for ind in range(len(valid_init_states)):
            trajectory_feat_exp = np.zeros(dim_features+25)
            curr_g = 1.0
            curr_pol = -1 * np.ones(nclusters + 1,dtype=int)
            for i in range(trajectory_length[valid_inds[ind]]):
                ext_feat = np.zeros(26)
                ext_feat[valid_trajectory_actions[ind][i] + 1] = 1.0
                curr_st = valid_trajectory_states[ind][i]
                trajectory_feat_exp += curr_g * np.append(Kmclusters[curr_st],ext_feat)
                curr_g *= gamma
                curr_pol[curr_st] = valid_trajectory_actions[ind][i]
                bc_input_valid.append(np.append(valid_contexts[ind],Cpca[valid_cont_dyn[ind]][curr_st]))
                bc_output_valid.append(ext_feat[1:])
        if RUN_TEST:
            bc_eval_in = []
            bc_eval_out = []
            policies_expert = []
            test_expert_value = []
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
                    pol_feat = np.zeros(25)
                    pol_feat[test_trajectory_actions[jk][i]] = 1.0
                    bc_eval_out.append(pol_feat)
                    bc_eval_in.append(np.append(testset[jk],Cpca[test_cont_dyn[jk]][curr_st]))
                policies_expert.append(curr_pol)
                features_expert_z.append(trajectory_feat_exp)
        bc_eval_in = np.asarray(bc_eval_in)
        bc_eval_out = np.asarray(bc_eval_out)
        train_rand_inds = np.arange(len(train_init_states))
        for itrainset in range(1):
            save_obj(d, "bc_values_5"+str(valuesindex))
            random.shuffle(train_rand_inds)
            Conts = train_contexts[train_rand_inds].copy()
            Tlen = trajectory_length[train_inds][train_rand_inds]
            testTlen = trajectory_length[test_inds]
            Conts_died = died_train[train_rand_inds].copy()
            Conts_dyn = train_cont_dynamics[train_rand_inds].copy()
            Train_features = np.asarray(training_featuresss)[train_rand_inds].copy()
            Train_policies = np.asarray(train_expert_policy)[train_rand_inds].copy()
            train_inits = np.asarray(train_init_states)[train_rand_inds].copy()

            bc_input_i = []
            bc_output_i = []
            for indd in train_rand_inds:
                bc_input_i.append(bc_inputs[indd])
                bc_output_i.append(bc_outputs[indd])

            accuracy = []
            action_dist = []

            for num_contexts in [1]:
                bc_input_shuff = []
                bc_output_shuff = []

                bc_input_shuff = bc_input_i.copy()
                bc_output_shuff = bc_output_i.copy()
                reset_weights(m)
                ttlen = Tlen
                contexts = Conts
                training_inits = train_inits
                contexts_dynamics = Conts_dyn
                training_features = Train_features
                training_policy = Train_policies
                d[trainset, "train_accuracy", num_contexts] = []
                es = EarlyStopping(monitor='val_loss', mode='min', patience=5, verbose=1)
                m.fit(x=np.asarray(bc_inputs), y=np.asarray(bc_outputs), batch_size=32,
                                    epochs=100,validation_data=(np.asarray(bc_input_valid),np.asarray(bc_output_valid)),callbacks=[es])
                pred_eval_out = m.predict(bc_eval_in)
                accur_ar = []
                for bbb in range(len(bc_eval_out)):
                    nums1 = np.arange(np.arange(25)[bc_eval_out[bbb].astype(bool)] % 5, 25,5).astype(np.int)
                    nums2 = np.arange(np.arange(25)[bc_eval_out[bbb].astype(bool)] - (np.arange(25)[bc_eval_out[bbb].astype(bool)] % 5),np.arange(25)[bc_eval_out[bbb].astype(bool)] - (np.arange(25)[bc_eval_out[bbb].astype(bool)] % 5) + 5).astype(np.int)
                    accur = pred_eval_out[bbb][nums2].sum()
                    accur_ar.append(accur)
                accuracy = np.asarray(accur_ar).mean()
                d[trainset, "train_accuracy", num_contexts].append(accuracy)
                print("===================== Evaluation:")
                print(" == Generalization for ",str(num_contexts), ", accuracy: ",str(accuracy))


                save_obj(d, "bc_values"+str(valuesindex))
                agent_vals = []
                agent_accurs = []
                test_agent_accuracy_ar = []



                save_obj(d, "bc_values"+str(valuesindex))


    save_obj(d, "bc_values"+str(valuesindex))
