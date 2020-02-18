#######################################################################################################################
#   file name: accuracy
#
#######################################################################################################################
# imports:
#######################################################################################################################
import numpy as np
#######################################################################################################################
# calculates accuracy given expert and agent policies:
def accuracy_mesure(agent_poicy, expert_policy,policy_size = 0):
    actions_diff = agent_poicy - expert_policy
    diffs = np.count_nonzero(actions_diff)

    if policy_size == 0:
        same = actions_diff.size - diffs
        return float(same)/float(actions_diff.size)
    else:
        same = policy_size - diffs
        return float(same)/float(policy_size)
