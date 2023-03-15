import matplotlib.pyplot as plt
import numpy as np

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

def generate_bids(users: list[User], rps: list[ResourceProvider], low: float=1.5, high: float=2.0, α = 5, β = 1e-9, ρ = lambda f: f*1e-9):
    """ There wasn't a description for how bids are generated.
    We are generating bids by:
    1. assigning initial RP bandwidths and frequencies to users
    2. sorting the RPs from low to high based on utility to the user
    3. assigning multipliers in a gausian distribution between low and high based on rank
    4. assigning bids based on cost to the RP
    
    Note: modifies rp bids """
    from algorithms import O_ji, C_ji

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
    range = high - low
    mid = low + range / 2
    std = range / 4
    multipliers = np.random.normal(size=len(rps), loc=mid, scale=std)
    for idx, i in enumerate(rps_sorted):
        i.m = max(multipliers[idx], 1.0)
    
    # 4. assign bids
    for i in rps:
        for j in users:
            i.b[j] = i.m * C_ji(j, i)

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
            valid_values = []
            for v in distribution:
                while v < minv or v > maxv:
                    v = np.random.normal(size=1, loc=loc, scale=scale)[0]
                valid_values.append(v)
            distribution = valid_values

        ret.append(distribution)
    
    return ret

def gen_users(n: int, plot=False):
    ret: list[User] = []
    
    fs, Cys, ds, Ps = gen_norms(n, [
        [2e9, 0.5e9, 0.1e9, 10e9], # (f)requency
        [2e12, 1e12, 0.2e12, 1e15], # (Cy)cles task requires
        [200e6, 200e6, 10e6, 1e12], # (d)ata bytes task requires
        [2, 1, 0.1, 4] # (P)ower to transmit with
    ])
    for i in range(n):
        ret.append(User(fs[i], Cys[i], ds[i], Ps[i], η=0.5, U_0=0))
    
    if plot:
        fig, ax = plt.subplots(2, 2)
        user_attributes = {
            "Frequency (GHz)": np.array(fs)/1e9,
            "Cycles (THz)": np.array(Cys)/1e12,
            "Bytes (MBs)": np.array(ds)/1e6,
            "Transmission Power (dBm)": Ps,
        }
        plot_idx = 0
        for title in user_attributes.keys():
            x, y = int(plot_idx/2), plot_idx%2
            attr = user_attributes[title]
            ax[x][y].hist( attr, np.linspace(min(attr),max(attr),int(n/5)) )
            ax[x][y].set_title(title)
            plot_idx += 1
        plt.show()

    return ret

def gen_resource_providers(all_users: list[User], n: int, f_tots: list[float]=None, W_tots: list[float]=None, plot=False):
    ret: list[ResourceProvider] = []
    
    if f_tots == None and W_tots == None:
        f_tots, W_tots = gen_norms(n, [
            [3e9, 3e9, 0.8e9, 10e9], # (f)requency available to this provider
            [5e6, 20e6, 1e6, 100e6], # (W)andwidth available to this provider
        ])
    for i in range(n):
        ret.append(ResourceProvider(all_users, f_tot=f_tots[i], W_tot=W_tots[i]))
    
    if plot:
        rp_attributes = {
            "Frequency (GHz)": (f_tots, 3e9+3e9*2, 9),
            "Bandwidth (MBs)": (W_tots, 5e6+20e6*2, 6),
        }
        if n < 10:
            fig, ax = plt.subplots()
            offset, width = 0, 0.25
            for title in rp_attributes.keys():
                attr, attr_max, e = rp_attributes[title]
                div_val = 10**e
                plot_attr = np.array(attr)/attr_max
                rects = ax.bar(np.array(range(n)) + offset, plot_attr, width, label=title)
                ax.bar_label(rects, ["%.2F" % (v/div_val) for v in attr])
                offset += width
            ax.legend()
            plt.show()
        else:
            fig, ax = plt.subplots(2, 1)
            plot_idx = 0
            for title in rp_attributes.keys():
                x = plot_idx % 2
                attr, attr_max, e = rp_attributes[title]
                plot_attr = np.array(attr)/attr_max
                ax[x].hist( plot_attr, np.linspace(min(plot_attr),max(plot_attr),int(n/5)) )
                ax[x].set_title(title)
                plot_idx += 1
        plt.show()
    
    return ret