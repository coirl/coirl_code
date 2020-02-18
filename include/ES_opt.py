#######################################################################################################################
#   file name: ES_opt
#
#   description:
#   this file defines the NES optimization method, used in the blackbox method.
#######################################################################################################################
# imports:
#######################################################################################################################
import numpy as np

#######################################################################################################################

class Result:
    def __init__(self,theta,val,iter,done):
        self.x = theta
        self.fun = val
        self.nfev = iter
        self.done = done


#######################################################################################################################
# ES method, implemented for solving CIRL problems:
def ES_minimize(target_func, init_step, sigma, num_eps, theta_init, maxiter, tol_stop=1e-10 ):

    # initiate current point and calculate the value on it:
    theta_t = theta_init
    theta_min = theta_t
    val = target_func(theta_t)

    # check if the value is below the pre-defined threshold:
    if val < tol_stop:
        return Result(theta_init,val,0,True)

    # set the current value to be the minimum:
    min = val

    # run maxiter iterations:
    for t in range(maxiter):

        weighted_sum = 0

        # sample random noise and evaluate the objective function on the noised point:
        for i in range(int(num_eps/2)):
            eps = np.random.normal(size=theta_init.shape)

            # we always take the opposite (negative) noise:
            feval = target_func(theta_t + (sigma * eps))
            n_feval = target_func(theta_t - (sigma * eps))

            # calculate the direction:
            weighted_sum += feval*eps - n_feval*eps

        # normalize the direction:
        normalized_weighted_sum = weighted_sum/(np.linalg.norm(weighted_sum))

        # take a step and evaluate:
        theta_t += (-1)*init_step*normalized_weighted_sum
        new_val = target_func(theta_t)

        # if the evaluation is better than the minimum we take the step:
        if new_val < min:
            min = new_val
            theta_min = theta_t

    return Result(theta_min, min, t, False)
#######################################################################################################################
