from algorithms import *
from classes import *

# testing exec_online_pmmra
def test_exec_online_pmmra():
    _all_users = gen_users(50, plot=False)
    _all_rps = gen_resource_providers(_all_users, 5, plot=False)
    generate_bids(_all_users, _all_rps)
    A = exec_online_pmmra(_all_users, _all_rps)
    print(A)

if __name__ == "__main__":
    test_exec_online_pmmra()