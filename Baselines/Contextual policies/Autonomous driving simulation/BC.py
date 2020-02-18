#######################################################################################################################
#   file name: BC
#
#   description:
#   this file builds a driving simulator environment.
#   then runs BC method to solve the linear COIRL problem.
#######################################################################################################################
# imports:
#######################################################################################################################
import sys
sys.path.append('../../../include')
import numpy as np
from ICMDP import *
from domains import *
import pickle
import random
from keras import optimizers
from keras.models import Model
from keras.layers import Input, Dense, Softmax
from keras.layers import LeakyReLU
from keras.callbacks import EarlyStopping
from keras import backend as K

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
tol = 1e-4
test_size = 80
num_contexts = 40
RUN_TEST = True
traj_nums = np.asarray([20,40,100,200,400,800,1200])

#######################################################################################################################
# Define BC model
#######################################################################################################################
def buildmodel():
    input1 = Input(shape=(803,))
    x = Dense(24,use_bias=False)(input1)
    x = LeakyReLU(alpha=0.1)(x)
    x = Dense(12,use_bias=False)(x)
    x = Dense(3,use_bias=False)(x)
    output1 = Softmax(axis=-1)(x)
    model = Model(inputs=[input1],outputs=[output1])
    sgd = optimizers.SGD(lr=0.1,decay=1e-7, momentum=0.9, nesterov=True)
    model.compile(loss='binary_crossentropy',optimizer=sgd)
    return model

def reset_weights(model):
    session = K.get_session()
    for layer in model.layers:
        if hasattr(layer, 'kernel_initializer'):
            layer.kernel.initializer.run(session=session)

#######################################################################################################################
# construct the CMDP
#######################################################################################################################
mdp = build_domain('DRV',0)

# load test set of contexts:
test_contexts = np.load("../../../data/autonomous_driving/all/test_set.npy")[:test_size]

# load real W:
real_W = np.load("../../../data/autonomous_driving/offline/realW.npy")

#######################################################################################################################
# Test the BC method in the environment
#######################################################################################################################
d = {}
m = buildmodel()

d["context_nums"] = np.asarray(traj_nums)

def feat_exp(r):
    return mdp.solve_MDP(gamma=0.9, tol=tol, w=r, flag='init')

def NL(x):
    return x @ real_W

test_expert_value = 0
test_expert_value_ar = []
policy_expert = []
for context in test_contexts:
    feat_expert_i = feat_exp(NL(context))
    policy_expert.append(feat_expert_i.policy)
    feat_expert = feat_expert_i.M
    test_expert_value_ar.append(NL(context) @ feat_expert)
test_expert_value = np.asarray(test_expert_value_ar).mean()
d["test_value_expert"] = test_expert_value
print("Expert test value: ", str(test_expert_value))
d["train_value_expert"] = []

# run seeds:
for trainset in range(repeats):
    if trainset > 0:
        save_obj(d, "bc_values" + str(valuesindex))

    for traj_len in traj_nums:
        num_contexts = 40
        reset_weights(m)
        run_i = trainset
        contexts = np.load("../../../data/autonomous_driving/all/train_set_"+str(trainset)+".npy")[:num_contexts]

        bc_input = []
        bc_output = []
        traj_states = []
        policy_expert_train = []
        train_expert_value_ar = []
        for traincont in contexts:
            feat_expert_i_train = feat_exp(NL(traincont))
            train_data_c = mdp.get_policy_for_train(feat_expert_i_train.policy,traincont,one_hot=0,pca_=800, init_state=-1,length=traj_len)
            bc_input.extend(train_data_c[0])
            bc_output.extend(train_data_c[1])
            traj_states.append(train_data_c[2])
            policy_expert_train.append(feat_expert_i_train.policy)
            train_expert_value_ar.append(NL(traincont) @ feat_expert_i_train.M)

        train_expert_value = np.asarray(train_expert_value_ar).mean()
        d["train_value_expert"].append(train_expert_value)
        print("train expert value: ", train_expert_value)

        if RUN_TEST:
            if trainset > 0:
                save_obj(d, "bc_values" + str(valuesindex))
            agent_vals = []
            agent_accurs = []
            comb_d = list(zip(bc_input,bc_output))
            random.shuffle(comb_d)
            bc_input_shuff, bc_output_shuff = zip(*comb_d)
            test_agent_value_ar = []
            test_agent_accuracy_ar = []
            es = EarlyStopping(monitor='val_loss',mode='min',patience=100,verbose=1)
            fit_hist = m.fit(x=np.asarray(bc_input_shuff),y=np.asarray(bc_output_shuff),batch_size=64,epochs=10000,validation_split=0.1,callbacks=[es])
            train_agent_value_ar = []
            train_agent_accuracy_ar = []

            for ci in range(len(contexts)):
                context = contexts[ci]
                policy_expert_i = policy_expert_train[ci]
                solution = mdp.feat_from_policy(0.9, context, m, tol=tol, flag='init',one_hot=0,pca_=800)
                train_agent_value_ar.append(NL(context) @ solution.M)

                train_agent_accuracy_ar.append(100.0 * np.asarray([solution.policy[ind,policy_expert_i[ind]] / float(len(traj_states[ci])) for ind in traj_states[ci]]).sum())

            train_agent_value = np.asarray(train_agent_value_ar).mean()
            train_agent_accuracy = np.asarray(train_agent_accuracy_ar).mean()
            print(" == Accuracy on train: ", str(train_agent_accuracy), "value on train: ", train_agent_value)
            d[trainset, "train_value", traj_len] = train_agent_value
            d[trainset, "train_accuracy", traj_len] = train_agent_accuracy
            for ci in range(len(test_contexts)):
                context = test_contexts[ci]
                policy_expert_i = policy_expert[ci]
                solution = mdp.feat_from_policy(0.9, context, m, tol=tol, flag='init',one_hot=0,pca_=800)
                test_agent_value_ar.append(NL(context) @ solution.M)
                test_agent_accuracy_ar.append(100.0 * np.asarray([solution.policy[ind,policy_expert_i[ind]] / float(len(policy_expert_i)) for ind in range(len(policy_expert_i))]).sum())

            test_agent_value = np.asarray(test_agent_value_ar).mean()
            test_agent_accuracy = np.asarray(test_agent_accuracy_ar).mean()
            agent_accurs.append(test_agent_accuracy)
            agent_vals.append(test_agent_value)
            print(" == Generalization for ", str(num_contexts), "contexts: ",
                  str(test_agent_value), ", runtime: ", ", accuracy: ", str(test_agent_accuracy))
            d[trainset, "test_value", traj_len] = agent_vals
            d[trainset, "accuracy", traj_len] = agent_accurs

save_obj(d, "bc_values" + str(valuesindex))

