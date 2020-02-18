import inspect
import os
from ICMDP import *
import scipy.io as sio
import numpy as np

curr_path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
##########################################################
# functions for Autonomous driving:
##########################################################
def do_action(state, action, min_speed, max_speed, min_x, max_x, step_size):
    speed = state.speed
    my_x = state.x

    # move left:
    if action == 1:
        if my_x - step_size >= min_x:
            my_x = my_x - step_size
        else:
            my_x = min_x

    # move right:
    elif action == 2:
        if my_x + step_size <= max_x:
            my_x = my_x + step_size
        else:
            my_x = max_x

    # increase speed:
    elif action == 3:
        if speed < max_speed:
            speed = speed + 1

    # decrease speed:
    elif action == 4:
        if speed > min_speed:
            speed = speed - 1

    return [speed, my_x]


# find next state function - finds the next possible states for a given state and action:
def find_next_state_drv(state, action, states_inv, other_car_x, speeds_num, bounds, height, displace, my_car_size):
    left_bound = bounds[0]
    right_bound = bounds[1]
    [new_speed, new_x] = do_action(state, action, speeds_num[0], speeds_num[-1], left_bound, right_bound, 5)

    # check if this is the first state:
    if (states_inv[','.join(str(elem) for elem in (state.as_list()))] == 0):

        # for first state - special control for speed, to choose the speed for the rest of the game:
        state_vec = []
        for x in other_car_x:
            if action == 0:
                init_speed = state.speed
            elif action == 1:
                init_speed = state.speed - 1
            elif action == 2:
                init_speed = state.speed + 1

            # insert a car in a random position:
            new_state = states_inv[','.join(str(elem) for elem in [init_speed, state.x, x, 10])]
            state_vec.append(new_state)
        return state_vec

    # check if need to insert a new car in a random place and remove the old one:
    elif (state.other_car[1] + displace[state.speed] >= height - 10 + my_car_size[0]):
        state_vec = []
        for x in other_car_x:
            new_state = states_inv[','.join(str(elem) for elem in [new_speed, new_x, x, 10])]
            state_vec.append(new_state)
        return state_vec

    # no new car needed - deterministic next state:
    else:
        new_state = states_inv[','.join(
            str(elem) for elem in [new_speed, new_x, state.other_car[0], state.other_car[1] + displace[state.speed]])]
        return new_state


##########################################################
def find_next_state_grid(state, action,nrow,ncol):
    # left bound:
    if ((state) % nrow == 0):
        # left
        if (action == 0):
            return state + nrow - 1
    # right bound
    if ((state + 1) % nrow == 0):
        # right
        if (action == 2):
            return state - nrow + 1
    # upper bound:
    if (state < nrow):
        # up
        if (action == 1):
            return state + nrow*(ncol - 1)
    # lower bound:
    if (state > nrow*(ncol - 1) - 1):
        # down
        if (action == 3):
            return state - nrow*(ncol - 1)
    # left
    if (action == 0):
        return state - 1
    # right
    if (action == 2):
        return state + 1
    # up
    if (action == 1):
        return state - nrow
    # down
    if (action == 3):
        return state + nrow
##################################################################

def build_domain(name, parameters):
    if name == 'DRV':
        # features:
        # 1. speed
        # 2. collisions
        # 3. off-road

        # actions:
        # 0 - do nothing
        # 1 -  move left
        # 2 - move right

        # parameters:
        # right-left step size:
        step_size = 5
        traj_len = 40
        # boundaries of the frame
        left_bound = 120
        right_bound = 200
        height = 180
        width = 300
        bottom_bound = height

        # boundaries of the road:
        road_left_bound = left_bound + 20
        road_right_bound = right_bound - 20

        # car size, width is half of the width in the format "[length,width]":
        my_car_size = [40, 10]

        # the y position of the player's car (stays fixed during the game):
        my_y = height - 10 - my_car_size[0]

        # initiate the speed feature values, displace for each speed and numbering:
        displace = [20, 80, 120]
        # displace = [20, 40, 80]
        speeds_num = [0, 1, 2]
        speed_feature_vals = [0.5, 0.75, 1]

        # calculate the different possible x positions of the player's car:
        my_x = []
        for x in range(left_bound, right_bound + step_size, step_size):
            my_x.append(x)

        # the lanes locations:
        lanes = [140, 160, 180]  # the x coordinates of the lanes

        # build other_car:
        other_car_length = 40
        other_car_width = 5
        other_car_x = lanes  # to lower complexity
        other_car_y = []  # the legal y coordinates of the other cars
        for i in range(10):
            other_car_y.append(20 * i + 10)

        other_car = []  # format: [x coordinate, y coordinate]
        for x in other_car_x:
            for y in other_car_y:
                other_car.append([x, y])

        # build actions:
        # 0 - do nothing
        # 1 - move left
        # 2 - move right
        actions = [0, 1, 2]

        # initiate states array and state to index (states_inv) dictionary:
        states = []
        states_inv = {}

        # initiate features:
        F = Features(dim_features=3)

        # add first  state:
        states.append(State(1, 160, [-1, -1]))
        states_inv[','.join(str(elem) for elem in (states[0].as_list()))] = 0
        F.add_feature(feature=[0.75, 0.5, 0.5])

        # build the whole state - feature mapping:
        for speed in speeds_num:
            for x in my_x:
                for other_x in other_car_x:
                    for other_y in other_car_y:
                        states.append(State(speed, x, [other_x, other_y]))
                        states_inv[','.join(str(elem) for elem in (states[len(states) - 1].as_list()))] = len(
                            states) - 1

                        # add speed feature value:
                        speed_val = speed_feature_vals[speed]

                        # check collision:
                        if (other_y > my_y) and (other_y - other_car_length < my_y + my_car_size[0]) and (
                                other_x + other_car_width > x - my_car_size[1]) and (
                                other_x - other_car_width < x + my_car_size[1]):
                            collision_val = 0.5
                        else:
                            collision_val = 0

                        # check off-road:
                        if (x < road_left_bound) or (x > road_right_bound):
                            off_road_val = 0.5
                        else:
                            off_road_val = 0

                        F.add_feature(feature=[speed_val, collision_val, off_road_val])

        # setup transitions:
        THETA = Transitions(num_states=len(states), num_actions=len(actions))
        curr_state = 0
        for state in states:
            for action in actions:

                # find next state:
                new_state = find_next_state_drv(state, action, states_inv, other_car_x, speeds_num, [left_bound,right_bound],height,displace,my_car_size)

                # if there is more than 1 possible next state, calculate uniform distribution between the possibilities:
                if isinstance(new_state, list):
                    num_states = len(new_state)
                    trans = 1.0 / num_states
                    for i in range(num_states):
                        THETA.set_trans(curr_state, action, new_state[i], trans)

                # deterministic next state:
                else:
                    THETA.set_trans(curr_state, action, new_state, 1)

            curr_state = curr_state + 1
    elif name == 'MED':

        make_pos = parameters

        data = sio.loadmat(curr_path + '/../data/dynamic_treatment/all/normalized_data.mat')

        ntraj = len(data['normalized_data'])
        r = []
        a = []
        phi = []
        m = []
        state_features = []
        traj_len = []

        # define some parameters
        nclusters = 500  # amount clusters in the static context domain
        n_static = 8  # 8-amount of static features to use for lstd
        nActions = 25
        jump = 3

        # get data
        expert_contexts = []
        for i in range(ntraj):
            r.append(data['normalized_data'][i][0][0][0][2][0])
            m.append(data['normalized_data'][i][0][0][0][6][0])
            a.append(data['normalized_data'][i][0][0][0][4])
            phi.append(data['normalized_data'][i][0][0][0][3])
            traj_len.append(phi[i].shape[0])
            non_norm_context = phi[i][0, 0:n_static] + 1
            expert_contexts.append(non_norm_context / np.sum(non_norm_context))

        for i in range(ntraj):
            j = 0
            while j < traj_len[i]:
                state_features.append(phi[i][j, n_static:])
                j += jump  # 1

        Kmclusters = np.load(curr_path + "/../data/dynamic_treatment/all/Kmeans_clusters.npy")
        Kmlabels = np.load(curr_path + "/../data/dynamic_treatment/all/Kmeans_labels.npy")
        dim_features = len(phi[0][0, n_static:]) + 1
        F = Features(dim_features=dim_features)
        feature = np.zeros(dim_features)
        for i in range(nclusters):
            feature[:-1] = Kmclusters[i, :]
            if make_pos:
                feature = (feature + 1.0) / 2.0
            feature[-1] = 0
            F.add_feature(feature=feature)
        # add features of 2 possible final states:
        feature = np.zeros(dim_features)
        F.add_feature(feature=feature)
        F.add_feature(feature=feature)

        feature[-1] = 1
        F.add_feature(feature=feature)

        transitions = []

        for i in range(nclusters + 2):
            transitions.append([])
            for j in range(nActions):
                transitions[i].append([])
                for k in range(nclusters + 2):
                    transitions[i][j].append(0)

        tot_traj = 0
        for i in range(ntraj):
            j = 0
            trajectory = []
            while jump * j < traj_len[i]:
                trajectory.append(Kmlabels[tot_traj + j])
                j += 1
            tot_traj += int(traj_len[i] / jump)

            for k in range(int(traj_len[i] / jump) - 1):
                transitions[trajectory[k]][a[i][k][0] - 1][trajectory[k + 1]] += 1

        THETA = Transitions(num_states=nclusters + 3, num_actions=nActions)

        for i in range(nclusters):
            for j in range(nActions):
                sum_trans = 0
                for k in range(nclusters + 2):
                    sum_trans += transitions[i][j][k]
                if (sum_trans == 0):
                    THETA.set_trans(i, j, nclusters + 2, 1)
                else:
                    for k in range(nclusters + 2):
                        prob = float(transitions[i][j][k]) / float(sum_trans)
                        THETA.set_trans(i, j, k, prob)
        for i in range(nActions):
            THETA.set_trans(nclusters, i, nclusters, 1)
            THETA.set_trans(nclusters + 1, i, nclusters + 1, 1)
            THETA.set_trans(nclusters + 2, i, nclusters + 2, 1)

    elif name == 'GRID':
        dims = parameters[:2]
        states = np.arange(dims[0] * dims[1])
        actions = [0, 1, 2, 3]
        if (len(parameters) > 2 and parameters[2]):
            num_contexts = parameters[3].shape[0]
            dim_contexts = parameters[3].shape[1]
            contexts = parameters[3]
            F = Features(dim_features=len(states) * dim_contexts)
            # add features:
            for ct in contexts:
                for st in states:
                    feat = np.zeros(len(states))
                    feat[st] = 1
                    F.add_feature(feature=(np.outer(ct, feat)).flatten())

            # build the whole state - feature mapping:

            # setup transitions:
            THETA = Transitions(num_states=len(states) * num_contexts, num_actions=len(actions))
            for ct_i in range(num_contexts):
                for state in states:
                    for action in actions:
                        # find next state:
                        new_state = find_next_state_grid(state, action,dims[0],dims[1])

                        # deterministic next state:
                        THETA.set_trans(ct_i * len(states) + state, action, ct_i * len(states) + new_state, 1)

        else:
            F = Features(dim_features=len(states))
            # add features:
            for st in states:
                feat = np.zeros(len(states))
                feat[st] = 1
                F.add_feature(feature=feat)

            # build the whole state - feature mapping:
            # setup transitions:
            THETA = Transitions(num_states=len(states), num_actions=len(actions))
            for state in states:
                for action in actions:
                    # find next state:
                    new_state = find_next_state_grid(state, action, dims[0], dims[1])

                    # deterministic next state:
                    THETA.set_trans(state, action, new_state, 1)

    else:
        print("Illegal value for 'name': ", name, "\nShould be: 'DRV' / 'MED / 'GRID'.")
        exit(1)


    mdp = ICMDP()

    # set the calculated features and transitions:
    mdp.set_F(F)
    mdp.set_THETA(THETA)
    return mdp