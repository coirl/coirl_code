#######################################################################################################################
#   file name: BC_lin
#
#   description:
#   this file builds a grid-world environment.
#   then runs BC with linear mapping from context to policy to solve the linear COIRL problem.
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
import os
from keras import optimizers
from keras.models import Model
from keras.layers import Input, Dense, Softmax
from keras.callbacks import EarlyStopping
from keras import backend as K

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
test_size = 100
context_nums = np.asarray([100,200,300,400,500])
dims_vec = [(3,3),(4,4),(5,5),(6,6),(7,7),(8,8),(9,9),(10,10)]
repeats = 5
tol = 1e-5
valid_crit = 100
valid_split = 0.1
max_iters = 10000
batch_size = 64
RUN_TEST = True

#######################################################################################################################
# Helper functions
#######################################################################################################################
def reset_weights(model):
    session = K.get_session()
    for layer in model.layers:
        if hasattr(layer, 'kernel_initializer'):
            layer.kernel.initializer.run(session=session)

#######################################################################################################################
# Train & Evaluation
#######################################################################################################################
for dims in dims_vec:
    print("####################################################################")
    print("%%%%%%%%%%%%% running with dimensions: " + str(dims[0]) + "X" + str(dims[1]))
    print("####################################################################")

    d = {}
    d["context_nums"] = np.asarray(context_nums)
    states = np.arange(dims[0]*dims[1])
    dim_contexts = len(states)

    all_contexts = np.load("../../../data/grid_world/rand_contexts_" + str(dims[0]) + "X" + str(dims[1]) + ".npy")
    test_contexts = all_contexts[all_contexts.shape[0] - test_size: all_contexts.shape[0]]

    # the model defined by the problem dimensions:
    mdp = build_domain('GRID', [dims[0], dims[1], False])
    real_W = (np.eye(dim_contexts))
    real_W /= np.sum(real_W.flatten())

    def buildmodel():
        input1 = Input(shape=(len(states)*dim_contexts,))
        x = Dense(4, use_bias=False)(input1)
        output1 = Softmax(axis=-1)(x)
        model = Model(inputs=[input1], outputs=[output1])
        sgd = optimizers.SGD(lr=3.0, decay=1e-9, momentum=0.9, nesterov=True)
        model.compile(loss='binary_crossentropy', optimizer=sgd)
        return model

    def feat_exp(r):
        return mdp.solve_MDP(gamma=gamma, tol=tol, w=r, flag='uniform')

    def NL(x):
        return x @ real_W / np.linalg.norm(real_W.flatten(), np.inf)


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
    m = buildmodel()

    for trainset in range(repeats):
        for num_contexts in context_nums:
            reset_weights(m)
            run_i = trainset
            contexts = all_contexts[trainset*context_nums[-1]:trainset*context_nums[-1] + num_contexts]
            bc_input = []
            bc_output = []
            policy_expert_train = []
            train_expert_value_ar = []

            for traincont in contexts:
                feat_expert_i_train = feat_exp(NL(traincont))
                train_data_c = mdp.get_policy_for_train(feat_expert_i_train.policy,traincont)
                bc_input.extend(train_data_c[0])
                bc_output.extend(train_data_c[1])
                policy_expert_train.append(feat_expert_i_train.policy)
                train_expert_value_ar.append(NL(traincont) @ feat_expert_i_train.M)

            train_expert_value = np.asarray(train_expert_value_ar).mean()
            d["train_value_expert"].append(train_expert_value)
            print("train expert value: ", train_expert_value)

            if RUN_TEST:
                if trainset > 0:
                    save_obj(d, "bc_lin_values" + str(valuesindex))
                agent_vals = []
                agent_accurs = []
                comb_d = list(zip(bc_input,bc_output))
                random.shuffle(comb_d)
                bc_input_shuff, bc_output_shuff = zip(*comb_d)
                test_agent_value_ar = []
                test_agent_accuracy_ar = []
                es = EarlyStopping(monitor='val_loss',mode='min',patience=valid_crit,verbose=1)
                fit_hist = m.fit(x=np.asarray(bc_input_shuff),y=np.asarray(bc_output_shuff),batch_size=batch_size,epochs=max_iters,validation_split=valid_split,callbacks=[es])
                train_agent_value_ar = []
                train_agent_accuracy_ar = []

                for ci in range(len(contexts)):
                    context = contexts[ci]
                    policy_expert_i = policy_expert_train[ci]
                    solution = mdp.feat_from_policy(gamma, context, m, tol=tol, flag='uniform')
                    train_agent_value_ar.append(NL(context) @ solution.M)
                    train_agent_accuracy_ar.append(100.0 * np.asarray([solution.policy[ind,policy_expert_i[ind]] / float(len(policy_expert_i)) for ind in range(len(policy_expert_i))]).sum())

                train_agent_value = np.asarray(train_agent_value_ar).mean()
                train_agent_accuracy = np.asarray(train_agent_accuracy_ar).mean()
                print(" == Accuracy on train: ", str(train_agent_accuracy), "value on train: ", train_agent_value)
                d[trainset, "train_value", num_contexts] = train_agent_value
                d[trainset, "train_accuracy", num_contexts] = train_agent_accuracy

                for ci in range(len(test_contexts)):
                    context = test_contexts[ci]
                    policy_expert_i = policy_expert[ci]
                    solution = mdp.feat_from_policy(gamma, context, m, tol=tol, flag='uniform')
                    test_agent_value_ar.append(NL(context) @ solution.M)
                    test_agent_accuracy_ar.append(100.0 * np.asarray([solution.policy[ind,policy_expert_i[ind]] / float(len(policy_expert_i)) for ind in range(len(policy_expert_i))]).sum())

                test_agent_value = np.asarray(test_agent_value_ar).mean()
                test_agent_accuracy = np.asarray(test_agent_accuracy_ar).mean()
                agent_accurs.append(test_agent_accuracy)
                agent_vals.append(test_agent_value)
                print(" == Generalization for ", str(num_contexts), "contexts: ",
                      str(test_agent_value), ", accuracy: ", str(test_agent_accuracy))
                d[trainset, "test_value", num_contexts] = agent_vals
                d[trainset, "accuracy", num_contexts] = agent_accurs

    save_obj(d, "bc_lin_values" + str(valuesindex))
    valuesindex += 1
