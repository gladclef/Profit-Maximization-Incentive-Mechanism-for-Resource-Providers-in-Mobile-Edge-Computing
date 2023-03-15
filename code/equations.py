import math
import numpy as np
from typing import Callable

from classes import *

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

# # testing find_optimal_rp_f
# u = User(f=1e9, Cy=200e9, d=200e6, P=2, η=0.5, U_0=0)
# r = ResourceProvider(f_tot=5.0e9)
# find_optimal_rp_f(j=u, i=r, α=5, β=1e-9, ρ=lambda f: f*1e-9, verbose=True)

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