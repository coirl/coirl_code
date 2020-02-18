#######################################################################################################################
#   file name: NON_LINEAR
#
#   description:
#   this file builds a medical environment from the MIMICIII normalized data,
#   where the real mapping from the context space to the reward is non-linear.
#   then runs the PSGD method to solve the non-linear COIRL problem.
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
from keras.models import Model
from keras.layers import Input, Dense, BatchNormalization
from keras.layers import LeakyReLU, Lambda
from keras import backend as K
from keras import regularizers
from keras import optimizers

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
trajectory_length = 40
iters = 100
repeats = 5
tol = 1e-3
batch_size = 32
test_size = 300
RUN_TEST = True

#######################################################################################################################
# construct the CMDP
#######################################################################################################################

mdp = build_domain('MED',False)

# load real W:
real_W_1 = np.load("../../data/dynamic_treatment/offline/real_W_1.npy")
real_W_2 = np.load("../../data/dynamic_treatment/offline/real_W_2.npy")
real_W_3 = np.load("../../data/dynamic_treatment/offline/real_W_3.npy")
real_W_4 = np.load("../../data/dynamic_treatment/offline/real_W_4.npy")
real_W_5 = np.load("../../data/dynamic_treatment/offline/real_W_5.npy")
real_W_6 = np.load("../../data/dynamic_treatment/offline/real_W_6.npy")

dim_context = real_W_1.shape[0]
dim_features = real_W_1.shape[1]

# load test and train sets of contexts and initial states:
testset = np.load("../../data/dynamic_treatment/all/testset.npy")[:test_size]
test_init_states = np.load("../../data/dynamic_treatment/all/test_init_states.npy")[:test_size]
train_contexts = np.load("../../data/dynamic_treatment/all/trainset.npy")
train_init_states = np.load("../../data/dynamic_treatment/all/train_init_states.npy")

#######################################################################################################################
# define functions for non-linear PSGD
#######################################################################################################################

def feat_exp(r,init_state):
    return mdp.solve_MDP(gamma=gamma,tol=tol,w=r,flag = 'init',init_state=init_state)

# The real mapping:
def NL(x):
    W = np.zeros(real_W_1.shape)
    if x[0] > 0.1:
        W += real_W_1
    else:
        W += real_W_2
    if x[2] > 0.11:
        W += real_W_3
    else:
        W += real_W_4
    if x[7] > 0.1:
        W += real_W_5
    else:
        W += real_W_6
    W /= np.linalg.norm(np.reshape(W,W.size),np.inf)
    return(W)

def custom_loss():
    def loss(y_true, y_pred):
        return K.batch_dot(y_pred,y_true,axes=1)
    return loss

def reset_weights(model):
    session = K.get_session()
    for layer in model.layers:
        if hasattr(layer, 'kernel_initializer'):
            layer.kernel.initializer.run(session=session)

def buildmodel():
    l = 0.001
    input1  = Input(shape=(dim_context,))
    x = Dense(8*dim_features,use_bias=True, kernel_regularizer = regularizers.l2(l=l))(input1)
    x = LeakyReLU(alpha=0.1)(x)
    x = BatchNormalization()(x)
    x = Dense(8*dim_features,use_bias=True, kernel_regularizer = regularizers.l2(l=l))(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = BatchNormalization()(x)
    
    x = Dense(8*dim_features,use_bias=True, kernel_regularizer = regularizers.l2(l=l))(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = BatchNormalization()(x)
    x = Dense(dim_features,use_bias=False, kernel_regularizer = regularizers.l2(l=l))(x)
    output1 = Lambda(lambda  a: K.l2_normalize(a+1e-10,axis=1))(x)
    model = Model(inputs=[input1],outputs=[output1])
    myopt = optimizers.SGD(clipnorm=1.5)
    model.compile(loss=custom_loss(),optimizer=myopt)
    return model

def Update_estimator(model,training_contexts_all,init_states_all, expert_feat_exp_all, max_iter=120, batch_size=1,
                     lr_init=0.1,decay=0.95):
    for iteration in range(1,max_iter+1):

        # calculate feature expectations for current parameters, for all contexts in train set
        agent_feat_exp_all = model.predict(training_contexts_all)

        # select batch
        indices = np.random.choice(a=list(range(len(training_contexts_all))), size=batch_size, replace=False)
        training_contexts = training_contexts_all[indices]
        expert_feat_exp = expert_feat_exp_all[indices]
        init_states = init_states_all[indices]
        agent_feat_exp = agent_feat_exp_all[indices]

        for aa in range(len(agent_feat_exp)):
            agent_feat_exp[aa] = np.array(feat_exp(agent_feat_exp[aa],init_states[aa]).M)

        feat_exp_diff = agent_feat_exp - expert_feat_exp

        # Take step
        LEARNING_RATE = lr_init * (decay ** iteration)
        K.set_value(model.optimizer.lr, LEARNING_RATE)
        model.fit(training_contexts, feat_exp_diff, batch_size=batch_size, epochs=1, verbose = 0)

    return model
#######################################################################################################################
# Test the PSGD method in the environment
#######################################################################################################################
d={}

m = buildmodel()

training_contexts = []
training_features = []
training_inits = []
for ind in range(len(train_init_states)):
    print(ind)
    training_contexts.append(train_contexts[ind])
    training_inits.append(train_init_states[ind])
    training_features.append(mdp.solve_CMDP(gamma=gamma, tol=tol, W=NL(train_contexts[ind]),context=train_contexts[ind], flag='init',init_state=train_init_states[ind],length=trajectory_length).M)

if RUN_TEST:
    policies_expert = []
    test_expert_value = []
    features_expert_z = []
    for jk in range(len(test_init_states)):
        features_expert = mdp.solve_CMDP(gamma=gamma, tol=tol, W=NL(testset[jk]), context=testset[jk], flag='init',init_state=test_init_states[jk])
        policies_expert.append(features_expert.policy)
        features_expert_z.append(features_expert.M)
        value_expert = ((1-gamma)/real_W_1.shape[1]) * testset[jk] @ NL(testset[jk]) @ features_expert.M
        test_expert_value.append(value_expert)
    test_expert_value = np.asarray(test_expert_value).mean()
    d["test_value"] = test_expert_value
    print("Expert test value: ",str(test_expert_value))

train_rand_inds = np.arange(len(train_init_states))
for trainset in range(repeats):
    save_obj(d, "psgd_values"+str(valuesindex))

    random.shuffle(train_rand_inds)
    reset_weights(m)
    train_inits = train_init_states[train_rand_inds].copy()
    train_inits = train_inits[:iters]
    expert_values = np.zeros(iters)
    agent_values = np.zeros(iters)
    context_count = 0
    ERR = False
    num_seen = 0
    accuracy = []
    action_dist = []

    for t in range(iters+1):
        if (RUN_TEST and (t % 5 == 0)):
            test_agent_value = []
            test_agent_loss = []
            accur = np.zeros(len(test_init_states))
            act_dist = np.zeros(len(test_init_states))
            for jk in range(len(test_init_states)):
                r_est = m.predict(np.expand_dims(testset[jk],axis=0))[0]
                features_agent_z = feat_exp(r_est, test_init_states[jk])
                loss_agent_z = ((1-gamma)/real_W_1.shape[1]) * r_est @ (features_agent_z.M - features_expert_z[jk])
                value_agent_z = ((1-gamma)/real_W_1.shape[1]) * testset[jk] @ NL(testset[jk]) @ (features_agent_z.M)
                test_agent_loss.append(abs(loss_agent_z))
                test_agent_value.append(abs(value_agent_z))
                accur[jk] = accuracy_mesure(features_agent_z.policy,policies_expert[jk])
                act_dist[jk] = actions_distance(features_agent_z.policy,policies_expert[jk])

            accuracy.append(accur.mean())
            test_agent_value = np.asarray(test_agent_value).mean()
            test_agent_loss = np.asarray(test_agent_loss).mean()
            d[trainset,"test_value",t] = test_agent_value
            d[trainset,"test_loss",t] = test_agent_loss
            print(" == Generalization for ",str(t), " iterations: \nValue: " + str(test_agent_value) + "\nLoss: " + str(test_agent_loss))
            print("Accuracy: ", accur.mean())

        rand_sample = random.choices(list(range(len(training_inits))),k=batch_size)
        training_contexts_curr = []
        training_inits_curr = []
        training_features_curr = []
        for sample in rand_sample:
            training_contexts_curr.append(training_contexts[sample])
            training_inits_curr.append(training_inits[sample])
            training_features_curr.append(training_features[sample])

        m = Update_estimator(model = m, training_contexts_all = np.array(training_contexts_curr),
                             init_states_all= np.asarray(training_inits_curr),
                             expert_feat_exp_all = np.array(training_features_curr),
                             max_iter = 1, batch_size = batch_size,
                             lr_init=0.3,decay=0.96**t)

    np.save("obj/accuracy_psgd_" + str(trainset) + ".npy",np.asarray(accuracy))
    d[trainset,"expert_values"] = expert_values
    d[trainset,"agent_values"] = agent_values

save_obj(d, "psgd_values"+str(valuesindex))
