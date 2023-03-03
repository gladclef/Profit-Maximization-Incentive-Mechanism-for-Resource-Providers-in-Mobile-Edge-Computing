import copy
import math
import numpy as np
from typing import Callable

from alt_name import *

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

all_users: list["User"] = []
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
        all_users.append(self)
    
    def __del__(self):
        all_users.remove(self)
    
    def __str__(self):
        return f"User(f={self.f},Cy={self.Cy},d={self.d},P={self.P},η={self.η},U_0={self.U_0})"

all_resource_providers: list["ResourceProvider"] = []
_next_resource_provider_idx = 0
class ResourceProvider():
    def __init__(self, f_tot, f: dict[User, float] = None, W: dict[User, float] = None, H: dict[User, float] = None, θ: dict[User, float] = None, ω: dict[User, float] = None):
        """ Args:
            f_tot (float): Total clock frequency available to this resource provider
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
        self.b: dict[User, float] = {j: 0 for j in all_users}
        """ Bids offered by this RP to each user """
        self.Π: dict[User, float] = {j: 0 for j in all_users}
        """ The profit, as calculated in algorithm 1 """
        all_resource_providers.append(self)

    def __del__(self):
        all_resource_providers.remove(self)
    
    def __str__(self):
        f_str = "[" + ",".join(self.f.values()) + "]"
        W_str = "[" + ",".join(self.W.values()) + "]"
        H_str = "[" + ",".join(self.H.values()) + "]"
        θ_str = "[" + ",".join(self.θ.values()) + "]"
        ω_str = "[" + ",".join(self.ω.values()) + "]"
        return f"ResourceProvider(f_tot={self.f_tot},f={f_str},W={W_str},H={H_str},θ={θ_str},ω={ω_str})"

@alt_name("eq1")
def completion_time_local(j: User):
    return j.Cy / j.f
T_jl = completion_time_local

@alt_name("eq2")
def energy_consumption_local(j: User):
    return κ * j.Cy * j.f**2
E_jl = energy_consumption_local

@alt_name("eq3")
def rate_task_offload(j: User, i: ResourceProvider) -> float:
    return i.W[j] * math.log2( 1 + (j.P * i.H[j] / N) )
r_ji = rate_task_offload

@alt_name("eq4")
def completion_time_remote(j: User, i: ResourceProvider) -> float:
    return j.d / r_ji(j, i) + j.Cy / i.f[j]
T_ji = completion_time_remote

@alt_name("eq5")
def energy_consumption_remote(j: User, i: ResourceProvider) -> float:
    return (j.d / r_ji(j,i)) * j.P   +   κ * j.Cy * i.f[j]**2
E_ji = energy_consumption_remote

@alt_name("eq7")
def rp_cost(j: User, i: ResourceProvider) -> float:
    # print(f"e:{p_e * κ * j.Cy * i.f[j]**2:.2E}, t:{p_t * j.Cy / i.f[j]:.2E}, w:{p_w * i.W[j]:.2E}")
    return p_e * κ * j.Cy * i.f[j]**2   +   p_t * j.Cy / i.f[j]   +   p_w * i.W[j]
C_ji = rp_cost

@alt_name("eq8")
def rp_revenue(j: User, i: ResourceProvider, α: float, β: float, ρ: Callable[[float], float]):
    """ Args:
        α (float): fixed connection price
        β (float): price charged per Hz used
        ρ (Callable[[float], float]): effect of demand changes on total frequency
    """
    return α + β*i.f[j] + ρ(i.f[j])
R_ji = rp_revenue

@alt_name("eq6")
def total_profit(j: User, i: ResourceProvider, α: float, β: float, ρ: Callable[[float], float]) -> float:
    return rp_revenue(j, i, α, β, ρ) - rp_cost(j, i)
Π_ji = total_profit

@alt_name("eq9")
def QoE(j: User, i: ResourceProvider, α: float, β: float, ρ: Callable[[float], float]) -> float:
    # print(f"e:{p_e * (E_jl(j) - E_ji(j, i)):.2E}, t:{p_t * (T_jl(j) - T_ji(j, i))}, Π:{j.η * rp_revenue(j, i, α, β, ρ)}")
    return p_e * (E_jl(j) - E_ji(j, i))   +   p_t * (T_jl(j) - T_ji(j, i))   -   j.η * rp_revenue(j, i, α, β, ρ)
O_ji = QoE

@alt_name("eq13")
def update_lagrange_1(j: User, i: ResourceProvider, k: int, α: float, β: float, ρ: Callable[[float], float]) -> float:
    """Args:
        k (int): Iteration index, assumed max of k_max
    """
    i.θ[j] = max(0, i.θ[j] + ε(k) * i.Π[j])
    return i.θ[j]
update_θ = update_lagrange_1

# @alt_name("eq13")
def update_lagrange_2(j: User, i: ResourceProvider, k: int, α: float, β: float, ρ: Callable[[float], float]) -> float:
    """Args:
        k (int): Iteration index, assumed max of k_max
    """
    i.ω[j] = max(0, i.ω[j] + ε(k) * ( QoE(j, i, α, β, ρ) - j.U_0 ))
    return i.ω[j]
update_ω = update_lagrange_2

@alt_name("eq12")
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

@alt_name("algorithm1")
def exec_non_competitive(α: float, β: float, ρ: Callable[[float], float]):
    users = copy.deepcopy(all_users)
    rps = copy.deepcopy(all_resource_providers)
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

            new_active_rps = copy.deepcopy(active_rps)
            for idx in active_rps:
                if rps[idx].θ[j] - old_θs[idx] <= τ:
                    new_active_rps.remove(idx)
            active_rps = new_active_rps

def a_ji(A: np.ndarray, j: User, i: ResourceProvider, new_val: int = None) -> int:
    if new_val != None:
        A[i.idx][j.idx] = new_val
    return A[i.idx][j.idx]

@alt_name("eq14")
def rp_utility(j: User, i: ResourceProvider, A: np.ndarray):
    return a_ji(A,j,i) * (i.p[j] - C_ji(j,i))
U_ji = rp_utility

@alt_name("eq16")
def bid_performance_ratio(j: User, i: ResourceProvider):
    return j.δ * i.b[j] / j.f   +   (1-j.δ) * j.d / r_ji(j, i) # TODO is j.f correct? should it be i.j?
BPR = bid_performance_ratio
γ_ji = bid_performance_ratio

@alt_name("eq17")
def price_performace_ratio(j: User, i: ResourceProvider):
    return i.p[j] / (i.f[j] - j.f)
PPR = price_performace_ratio
ζ_ji = price_performace_ratio

# @alt_name("algorithm2")
# def exec_online_pmmra():
#     eligible_users = copy.deepcopy(all_users)
#     rps = copy.deepcopy(all_resource_providers)
#     A = np.ndarray((len(rps),len(eligible_users)), order='F')
#     A.fill(0)
    
#     while len(eligible_users) > 0:
#         for j in eligible_users:


# testing find_optimal_rp_f
u = User(f=1e9, Cy=200e9, d=200e6, P=2, η=0.5, U_0=0)
r = ResourceProvider(f_tot=5.0e9)
find_optimal_rp_f(j=u, i=r, α=5, β=1e-9, ρ=lambda f: f*1e-9, verbose=True)