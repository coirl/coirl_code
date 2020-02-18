#######################################################################################################################
#   file name: NL_functions
#
#   description:
#   defines functions to be used in the nonlinear  ase
#######################################################################################################################
# imports:
#######################################################################################################################
import numpy as np
from ICMDP import *
#######################################################################################################################

LeakyReLU = lambda t: (0.45*abs(t)+0.55*t)

def feat_exp(r,mdp,tol):
    return mdp.solve_MDP(gamma=0.9,tol=tol,w=r,flag = 'init').M

def NL(x):
    if x @ np.array([1,0,0]) > 0.55 or x @ np.array([0,1,0]) > 0.55 or x @ np.array([0,0,1]) > 0.55 :
        return np.array([1.0,-1.0,-0.05])
    return np.array([-0.01,1.0,-1.0])

def NL_est(x,mat_list,activation=LeakyReLU,bias=0.0):
    y = x
    for i in range(len(mat_list)):
        y = np.concatenate((y,np.asarray([bias])))
        y = y @ mat_list[i]
        if i<len(mat_list)-1:
            y = activation(y)
        else:
            y = y/np.linalg.norm(y)
    return y

def evaluate(map_eval, training_contexts, expert_feat_exp, agent_feat_exp):
    return np.asarray([(NL_est(training_contexts[j],map_eval) @ (agent_feat_exp[j] - expert_feat_exp[j])) \
                       for j in range(len(training_contexts))]).sum()

def Update_estimator(mdp, tol, mat_list,training_contexts_all, expert_feat_exp_all, testset,max_iter=500,
                     stepsize=0.05, decay=0.99, std=0.1, num_pts=20, batch_size=1, epsilon=1e-3,
                     debug_interval = 50):
    # Create copy to not affect input matrices
    curr_list = [mat.copy() for mat in mat_list]

    # Initialize probability vector, make sure it is updated in 1st iteration
    probs = np.zeros(len(training_contexts_all))

    iteration = 1
    while iteration <= max_iter:
        agent_feat_exp_all = np.array([feat_exp(NL_est(training_context,curr_list),mdp,tol) for training_context in training_contexts_all])
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

        # Evaluate starting point for this minibatch. If minibatch is already close to optimal, go back and try again
        ev = evaluate(curr_list, training_contexts, expert_feat_exp, agent_feat_exp)

        #DEBUG PRINT
        if debug_interval:
            print("Iter ",iteration," out of ",max_iter)
        if debug_interval and iteration % debug_interval == 0:
            test_agent_value_test = np.asarray([((1-0.9)/3) * NL(ctest) @ feat_exp(NL_est(ctest,curr_list),mdp,tol) for ctest in testset]).mean()
            test_agent_value_train = np.asarray([((1-0.9)/3) * NL(ctest) @ feat_exp(NL_est(ctest,curr_list),mdp,tol) for ctest in training_contexts_all]).mean()
            print("Agent value test:",test_agent_value_test)
            print("Agent value train:",test_agent_value_train)


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
        step = [step[j] / np.linalg.norm(step[j]) for j in range(mat_num)]
        # Update current point
        curr_list = [curr_list[j] - stepsize*(decay**iteration)*step[j] for j in range(mat_num)]
        # If point is worse on this minibatch, cancel this step
        new_ev = evaluate(curr_list, training_contexts, expert_feat_exp, agent_feat_exp)

        if debug_interval and iteration % debug_interval == 0:
            print("Target function value, batch post: ", new_ev)
        iteration += 1

    return curr_list
