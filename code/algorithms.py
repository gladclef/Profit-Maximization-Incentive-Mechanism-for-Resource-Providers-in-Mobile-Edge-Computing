import copy
from functools import reduce
import math
import matplotlib.pyplot as plt
import numpy as np
from typing import Callable

from classes import *
from equations import *

# algorithm1
def exec_non_competitive(users: list[User], rps: list[ResourceProvider], α: float, β: float, ρ: Callable[[float], float]):
    active_rps = list(range(len(rps)))
    τ = 1e-100

    for j in users:
        for k in range(k_max):
            old_θs = [i.θ[j] for i in rps]

            for idx in active_rps:
                i = rps[idx]
                Π = Π_ji(j, i, α, β, ρ)
                O = O_ji(j, i, α, β, ρ)
                if Π > 0 and O - j.U_0 >= 0:
                    find_optimal_rp_f(j, i, α, β, ρ)
                    update_θ(j, i, k, α, β, ρ)
                    update_ω(j, i, k, α, β, ρ)
                    i.Π[j] = Π_ji(j, i, α, β, ρ)
                else:
                    i.Π[j] = 0

            new_active_rps = copy.copy(active_rps)
            for idx in active_rps:
                if rps[idx].θ[j] - old_θs[idx] <= τ:
                    new_active_rps.remove(idx)
            active_rps = new_active_rps

def bens_mod_filter_rps(rps: list[ResourceProvider], A: np.ndarray) -> list[ResourceProvider]:
    """ Filter out RPs that already have a match """
    ret: list[ResourceProvider] = []
    for rp in rps:
        if sum(A[rp.idx]) == 0:
            ret.append(rp)
    return ret

# algorithm2
def exec_online_pmmra(users: list[User], rps: list[ResourceProvider]) -> np.ndarray:
    """ Generate a matching matrix "A", given the users and resource providers to match.
    The Resource providers should come with bids already set.

    Note: modified rp bids

    Returns:
        np.ndarray: matrix of 0's and 1's, where matched RPs (index 0) and users (index 1)
                    are indicated by a 1.
    """
    rp_indexes = [rp.idx for rp in rps]
    user_indexes = [user.idx for user in users]
    A = np.ndarray( (max(rp_indexes)+1,max(user_indexes)+1), order='F' )
    A.fill(0)
    J = []
    
    # perform the auction
    while len(J) < len(users):
        # 0: filter out RPs that already have a match
        # we made this mod to what is in the paper so that other RPs have a chance to serve users
        eligible_rps = bens_mod_filter_rps(rps, A)
        if len(eligible_rps) == 0:
            break

        # 1: match RPs to users (lines 3-14)
        eligible_users = filter(lambda j: j not in J, copy.copy(users))
        for j in eligible_users:
            i_bprs = [(i, BPR(j,i)) for i in eligible_rps]
            i_bprs = list(filter(lambda i_bpr: i_bpr[0].b[j] != None, i_bprs))

            # find an RP with a bid that meets the constraints
            found_matching_rp = False
            while len(i_bprs) > 0:

                # find the min bpr
                i_bpr_min: tuple[ResourceProvider,float] = \
                    reduce(lambda v1, v2: v1 if v1[1] < v2[1] else v2, i_bprs)
                
                # verify constraints
                # C1: no more than one RP per user
                num_serving_rps = sum([ A[i2.idx][j.idx] for i2 in eligible_rps ])
                # check C1 (and ignore C2 <- I don't know how to check it)
                if (num_serving_rps > 0): # or (i.f[j] > i.f_tot or i.w[j] > i.W_tot):
                    i_bprs.remove(i_bpr_min)
                    i.b[j] = None # line 9: remove bid
                    continue
                
                # we found a minimum BPR that works!
                found_matching_rp = True
                break

            # assign the satisfying RP to the user
            if found_matching_rp:
                i = i_bpr_min[0]
                A[i.idx][j.idx] = 1
                other_rps = copy.copy(eligible_rps)
                other_rps.remove(i)
                if len(other_rps) > 0:
                    vickrey_bid = min([i2.b[j] for i2 in other_rps])
                    i.p[j] = vickrey_bid
                else:
                    i.p[j] = i.b[j]
            
            # this user can't be satisfied
            else:
                J.append(j)
        
        # 2: allow RPs to choose users (lines 15-21)
        for i in eligible_rps:
            # is this seller matched with more than one buyer?
            matched_users = list(filter(lambda j: A[i.idx][j.idx] == 1, users))
            if len(matched_users) <= 1:
                continue
            
            # find the max PPR
            j_pprs = [(j, PPR(j, i)) for j in matched_users]
            j_ppr_max: tuple[User,float] = \
                reduce(lambda v1, v2: v1 if v1[1] > v2[1] else v2, j_pprs)
            
            # choose the winner
            winner = j_ppr_max[0]
            J.append(winner)
            
            # put all other matched users in the looser group
            other_matched_users = filter(lambda j: j != winner, matched_users)
            for looser in other_matched_users:
                A[i.idx][looser.idx] = 0
                J.append(looser)
    
    return A