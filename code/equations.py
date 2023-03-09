import copy
from functools import reduce
import math
import numpy as np
from typing import Callable

κ = 10**-11
""" effective switched capacitance depending on chip architecture """
N = .1
""" the background noise power (Watts) """
# these values found expirimentally in order to make the math work
p_e = 1e-15
""" price of electricy (price/watt) """
p_ts = 100_00 / 3600
""" price of time (price/s) """
p_t = p_ts / 10e9
""" price of time (price/nanosecond) """
p_w = 1e-10
""" price of bandwidth (price/byte) """
k_max = 10
""" maximum number of iterations to find f_i in equation 12 """
ε = lambda k: math.log2( (k_max - k) / k_max + 1 ) * 0.5
""" step size per iteration """

_next_user_idx = 0
class User():
    def __init__(self, f: float = 0, Cy: float = 0, d: float = 0, P: float = 0, η: float = 0, U_0: float = 0, δ: float = 0):
        """ Args:
            f (float): Clock frequency of this user
            Cy (float): Number of cycles required for this user's task
            d (float): Data size of this user's task
            P (float): Transmission power
            η (float): Emphasize performance (0) or price (1), weight value between 0 and 1
            U_0 (float): Offloading benefit threshold
            δ (float): The risk factor, weight between 0 (stable transmission) and 1 (processing capacity)
        """
        global _next_user_idx
        self.idx = _next_user_idx
        """ Index of the User """
        _next_user_idx += 1
        self.f: float = f
        """ Clock frequency of this user """
        self.Cy: float = Cy
        """ Number of cycles required for this user's task """
        self.d: float = d
        """ Data size of this user's task """
        self.P: float = P
        """ Transmission power """
        self.η: float = η
        """ Emphasize performance (0) or price (1), weight value between 0 and 1 """
        self.U_0: float = U_0
        """ Offloading benefit threshold """
        self.δ: float = δ
        """ The risk factor, weight between 0 (stable transmission) and 1 (processing capacity) """
    
    def __str__(self):
        return f"User(f={self.f},Cy={self.Cy},d={self.d},P={self.P},η={self.η},U_0={self.U_0})"

_next_resource_provider_idx = 0
class ResourceProvider():
    def __init__(self, all_users: list[User], f_tot: float, W_tot: float, f: dict[User, float] = None, W: dict[User, float] = None, H: dict[User, float] = None, θ: dict[User, float] = None, ω: dict[User, float] = None):
        """ Args:
            f_tot (float): Total clock frequency available to this resource provider
            W_tot (float): Total bandwidth available to this resource provider
            f (dict[User, float]): Amount of f_tot that has been allocated to a given user
            W (dict[User, float]): Channel bandwidth provided by this RP to a given user
            H (dict[User, float]): Channel gain between this RP and a given user
            θ (dict[User, float]): Lagrange value 1, updated with equation 13
            ω (dict[User, float]): Lagrange value 2, updated with equation 13
        """
        global _next_resource_provider_idx
        self.idx = _next_resource_provider_idx
        """ Index of the ResourceProvider """
        _next_resource_provider_idx += 1
        self.f_tot: float = f_tot
        """ Total clock frequency available to this resource provider """
        self.W_tot: float = W_tot
        """ Total bandwidth available to this resource provider """
        self.f: dict[User, float] = f if f != None else {j: f_tot/5 for j in all_users}
        """ Amount of f_tot that has been allocated to a given user """
        self.W: dict[User, float] = W if W != None else {j: 5e6 for j in all_users}
        """ Channel bandwidth provided by this RP to a given user """
        self.H: dict[User, float] = H if H != None else {j: 1 for j in all_users}
        """ Channel gain between this RP and a given user """
        self.θ: dict[User, float] = θ if θ != None else {j: 0 for j in all_users}
        """ Lagrange value 1, updated with equation 13 """
        self.ω: dict[User, float] = ω if ω != None else {j: 0 for j in all_users}
        """ Lagrange value 2, updated with equation 13 """
        self.p: dict[User, float] = {j: 0 for j in all_users}
        """ Final payment offered to this RP by each user """
        self.b: dict[User, float|None] = {j: 0 for j in all_users}
        """ Bids offered by this RP to each user. None if the bid was removed by the rp during the second half of algo 2. """
        self.Π: dict[User, float] = {j: 0 for j in all_users}
        """ The profit, as calculated in algorithm 1 """
        self.m: float = 1
        """ Our custom multiplier for bids based on RP utility ranking. """

    def __str__(self):
        f_str = "[" + ",".join(self.f.values()) + "]"
        W_str = "[" + ",".join(self.W.values()) + "]"
        H_str = "[" + ",".join(self.H.values()) + "]"
        θ_str = "[" + ",".join(self.θ.values()) + "]"
        ω_str = "[" + ",".join(self.ω.values()) + "]"
        return f"ResourceProvider(f_tot={self.f_tot},f={f_str},W={W_str},H={H_str},θ={θ_str},ω={ω_str})"

# eq1
def completion_time_local(j: User):
    return j.Cy / j.f
T_jl = completion_time_local

# eq2
def energy_consumption_local(j: User):
    return κ * j.Cy * j.f**2
E_jl = energy_consumption_local

# eq3
def rate_task_offload(j: User, i: ResourceProvider) -> float:
    return i.W[j] * math.log2( 1 + (j.P * i.H[j] / N) )
r_ji = rate_task_offload

# eq4
def completion_time_remote(j: User, i: ResourceProvider) -> float:
    return j.d / r_ji(j, i) + j.Cy / i.f[j]
T_ji = completion_time_remote

# eq5
def energy_consumption_remote(j: User, i: ResourceProvider) -> float:
    return (j.d / r_ji(j,i)) * j.P   +   κ * j.Cy * i.f[j]**2
E_ji = energy_consumption_remote

# eq7
def rp_cost(j: User, i: ResourceProvider) -> float:
    # print(f"e:{p_e * κ * j.Cy * i.f[j]**2:.2E}, t:{p_t * j.Cy / i.f[j]:.2E}, w:{p_w * i.W[j]:.2E}")
    return p_e * κ * j.Cy * i.f[j]**2   +   p_t * j.Cy / i.f[j]   +   p_w * i.W[j]
C_ji = rp_cost

# eq8
def rp_revenue(j: User, i: ResourceProvider, α: float, β: float, ρ: Callable[[float], float]):
    """ Args:
        α (float): fixed connection price
        β (float): price charged per Hz used
        ρ (Callable[[float], float]): effect of demand changes on total frequency
    """
    return α + β*i.f[j] + ρ(i.f[j])
R_ji = rp_revenue

# eq6
def total_profit(j: User, i: ResourceProvider, α: float, β: float, ρ: Callable[[float], float]) -> float:
    return rp_revenue(j, i, α, β, ρ) - rp_cost(j, i)
Π_ji = total_profit

# eq9
def QoE(j: User, i: ResourceProvider, α: float, β: float, ρ: Callable[[float], float]) -> float:
    # print(f"e:{p_e * (E_jl(j) - E_ji(j, i)):.2E}, t:{p_t * (T_jl(j) - T_ji(j, i))}, Π:{j.η * rp_revenue(j, i, α, β, ρ)}")
    return p_e * (E_jl(j) - E_ji(j, i))   +   p_t * (T_jl(j) - T_ji(j, i))   -   j.η * rp_revenue(j, i, α, β, ρ)
O_ji = QoE

# eq13
def update_lagrange_1(j: User, i: ResourceProvider, k: int, α: float, β: float, ρ: Callable[[float], float]) -> float:
    """Args:
        k (int): Iteration index, assumed max of k_max
    """
    i.θ[j] = max(0, i.θ[j] + ε(k) * i.Π[j])
    return i.θ[j]
update_θ = update_lagrange_1

# # eq13
def update_lagrange_2(j: User, i: ResourceProvider, k: int, α: float, β: float, ρ: Callable[[float], float]) -> float:
    """Args:
        k (int): Iteration index, assumed max of k_max
    """
    i.ω[j] = max(0, i.ω[j] + ε(k) * ( QoE(j, i, α, β, ρ) - j.U_0 ))
    return i.ω[j]
update_ω = update_lagrange_2

# eq12
def find_optimal_rp_f(j: User, i: ResourceProvider, α: float, β: float, ρ: Callable[[float], float], verbose=False) -> float:
    X = p_e * κ * j.Cy
    Y = p_t * j.Cy
    γ1 = - (2*X*(1+i.θ[j]+i.ω[j])) / (β*(1+i.θ[j]-i.ω[j]*j.η))
    γ2 = 0
    γ3 = - Y / (2*X)
    m = 2/27 * γ1**3 + γ3
    l = -1/3 * γ1**2
    Δ = m**2 + (4 * l**3 / 27)
    Δ = max(Δ, 0)
    v = np.cbrt( (-m - math.sqrt(Δ)) / 2 )
    u = np.cbrt( (-m + math.sqrt(Δ)) / 2 )
    
    i.f[j] = u + v - 1/3*γ1
    i.f[j] = max(1, min(i.f_tot, i.f[j]))

    if verbose:
        print(f"f:{i.f[j]:.2E}, Π:{i.Π[j]:.2E}, θ:{i.θ[j]:.2E}, ω:{i.ω[j]:.2E}")
    
    return i.f[j]

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

def a_ji(A: np.ndarray, j: User, i: ResourceProvider, new_val: int = None) -> int:
    if new_val != None:
        A[i.idx][j.idx] = new_val
    return A[i.idx][j.idx]

# eq14
def rp_utility(j: User, i: ResourceProvider, A: np.ndarray):
    return a_ji(A,j,i) * (i.p[j] - C_ji(j,i))
U_ji = rp_utility

# eq16
def bid_performance_ratio(j: User, i: ResourceProvider):
    if i.b[j] == None:
        return None
    return j.δ * i.b[j] / j.f   +   (1-j.δ) * j.d / r_ji(j, i) # TODO is j.f correct? should it be i.j?
BPR = bid_performance_ratio
γ_ji = bid_performance_ratio

# eq17
def price_performace_ratio(j: User, i: ResourceProvider):
    return i.p[j] / (i.f[j] - j.f)
PPR = price_performace_ratio
ζ_ji = price_performace_ratio

# algorithm2
def exec_online_pmmra(users: list[User], rps: list[ResourceProvider]) -> np.ndarray:
    """ Generate a matching matrix "A", given the users and resource providers to match.
    The Resource providers should come with bids already set.

    Note: modified rp bids

    Returns:
        np.ndarray: matrix of 0's and 1's, where matched RPs (index 0) and users (index 1)
                    are indicated by a 1.
    """
    A = np.ndarray((len(rps),len(users)), order='F')
    A.fill(0)
    J = []
    
    # perform the auction
    while len(J) < len(users):
        # 1: match RPs to users (lines 3-14)
        eligible_users = filter(lambda j: j not in J, copy.copy(users))
        for j in eligible_users:
            i_bprs = [(i, BPR(j,i)) for i in rps]
            i_bprs = list(filter(lambda i_bpr: i_bpr[0].b[j] != None, i_bprs))

            # find an RP with a bid that meets the constraints
            found_matching_rp = False
            while len(i_bprs) > 0:

                # find the min bpr
                i_bpr_min: tuple[ResourceProvider,float] = \
                    reduce(lambda v1, v2: v1 if v1[1] < v2[1] else v2, i_bprs)
                
                # verify constraints
                # C1: no more than one RP per user
                num_serving_rps = sum([ A[i2.idx][j.idx] for i2 in rps ])
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
                other_rps = copy.copy(rps)
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
        for i in rps:
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

def generate_bids(users: list[User], rps: list[ResourceProvider], low: float=1.5, high: float=2.0, α = 5, β = 1e-9, ρ = lambda f: f*1e-9):
    """ There wasn't a description for how bids are generated.
    We are generating bids by:
    1. assigning initial RP bandwidths and frequencies to users
    2. sorting the RPs from low to high based on utility to the user
    3. assigning multipliers in a gausian distribution between low and high based on rank
    4. assigning bids based on cost to the RP
    
    Note: modifies rp bids """
    # 1. assign initial f and W
    for i in rps:
        for j in users:
            i.f[j] = i.f_tot
            i.W[j] = i.W_tot
    
    # 2. sort based on utility
    old_U_0 = {j: j.U_0 for j in users}
    for j in users:
        j.U_0 = 0
    rps_sorted = sorted(rps, key=lambda i: O_ji(j,i,α,β,ρ), reverse=True)
    for j in users:
        j.U_0 = old_U_0[j]

    # 3. assign multipliers
    mid = (high - low) / 2 + low
    std = (high - low) / 2
    multipliers = np.random.normal(size=len(rps), loc=mid, scale=std)
    for idx, i in enumerate(rps_sorted):
        i.m = max(multipliers[idx], 1.0)
    
    # 4. assign bids
    for i in rps:
        for j in users:
            i.b[j] = i.m * C_ji(j, i)

# # testing find_optimal_rp_f
# u = User(f=1e9, Cy=200e9, d=200e6, P=2, η=0.5, U_0=0)
# r = ResourceProvider(f_tot=5.0e9)
# find_optimal_rp_f(j=u, i=r, α=5, β=1e-9, ρ=lambda f: f*1e-9, verbose=True)

def gen_norms(count: int, params: list[tuple[float,float]|tuple[float,float,float,float]]):
    """ Calls np.random.normal(size, loc=params[i][0], scale=params[i][0]) size times.

    Args:
        params (list[tuple[float,float] | tuple[float,float,float,float]]): A list of (loc,scale[,min,max]) parameters for creating distributions.

    Returns:
        list[list[float]]: N distributions of count values, where N = len(params)
    """
    ret: list[list[float]] = []

    for params_i in params:
        loc = params_i[0]
        scale = params_i[1]
        distribution = np.random.normal(size=count, loc=loc, scale=scale)

        if len(params_i) > 2:
            minv = params_i[2]
            maxv = params_i[3]
            distribution = list(map(lambda v: min(max(v, minv), maxv), distribution))

        ret.append(distribution)
    
    return ret

def gen_users(n: int):
    ret: list[User] = []
    
    fs, Cys, ds, Ps = gen_norms(n, [
        [1e9, 1.0e9, 0.2e9, 10e9], # (f)requency
        [200e9, 100e9, 100e9, 1e12], # (Cy)cles task requires
        [200e6, 1e9, 10e6, 1e12], # (d)ata bytes task requires
        [2, 1, 1, 4] # (P)ower to transmit with
    ])
    for i in range(n):
        ret.append(User(fs[i], Cys[i], ds[i], Ps[i], η=0.5, U_0=0))
    
    return ret

def gen_resource_providers(all_users: list[User], n: int, f_tots: list[float]=None, W_tots: list[float]=None):
    ret: list[ResourceProvider] = []
    
    if f_tots == None and W_tots == None:
        f_tots, W_tots = gen_norms(n, [
            [3e9, 3e9, 0.8e9, 10e9], # (f)requency available to this provider
            [5e6, 20e6, 1e6, 100e6], # (W)andwidth available to this provider
        ])
    for i in range(n):
        ret.append(ResourceProvider(all_users, f_tot=f_tots[i], W_tot=W_tots[i]))
    
    return ret

# testing exec_online_pmmra
_all_users = gen_users(50)
_all_rps = gen_resource_providers(_all_users, 5)
arp_ftots = [rp.f_tot for rp in _all_rps]
arp_Wtots = [rp.W_tot for rp in _all_rps]
generate_bids(_all_users, _all_rps)
A = exec_online_pmmra(_all_users, _all_rps)
print(A)