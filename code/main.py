from algorithms import *
from classes import *

# testing exec_online_pmmra
def test_exec_online_pmmra():
    _all_users = gen_users(50, plot=False)
    _all_rps = gen_resource_providers(_all_users, 5, plot=False)
    generate_bids(_all_users, _all_rps)
    A = exec_online_pmmra(_all_users, _all_rps)
    print(A)

class Event():
    def __init__(self, tick_secs: float, tick: int, user: User, resource_provider: ResourceProvider):
        self.tick_secs = tick_secs
        """ How many seconds a tick represents """
        self.start_tick = tick
        """ The time slice that this event starts at """
        self.user = user
        self.rp = resource_provider

        self.upload_secs = self.user.d / self.rp.W[user]
        self.compute_secs = self.user.Cy / self.rp.f[user]
        self.download_secs = 0 #self.user.d / self.rp.W[user]
        self.total_secs = self.upload_secs + self.compute_secs + self.download_secs
        self.end_tick = int(math.ceil( self.start_tick + self.total_secs / self.tick_secs ))

class Auctioneer():
    def __init__(self, tick_secs: float, all_users: list[User], all_rps: list[ResourceProvider]):
        self.tick_secs = tick_secs
        self.all_users = all_users
        self.all_rps = all_rps
        self.tick = 0
        self.events: list[Event] = []

    def get_status(self, tick: int):
        satisfied_users: list[User] = []
        busy_rps: list[ResourceProvider] = []
        for event in self.events:
            if event.start_tick > tick:
                continue
            satisfied_users.append(event.user)
            if event.end_tick > tick:
                busy_rps.append(event.rp)
                
        available_users, available_rps = copy.copy(self.all_users), copy.copy(self.all_rps)
        for user in satisfied_users:
            available_users.remove(user)
        for rp in busy_rps:
            available_rps.remove(rp)
        
        return satisfied_users, busy_rps, available_users, available_rps

    def get_event_ticks(self) -> list[tuple[int,list[Event]]]:
        if len(self.events) == 0:
            return None
        
        events_list: list[tuple[int,list[Event]]] = []
        for event in self.events:
            events_list += [(event.start_tick, [event]), (event.end_tick, [event])]
        events_list.sort(key=lambda t_e: t_e[0])
        
        ret = [events_list[0]]
        for tick, events in events_list[1:]:
            event = events[0]
            if ret[-1][0] == tick:
                ret[-1][1].append(event)
            else:
                ret.append((tick, [event]))

        return ret

    def get_event(self, tick: int, users: list[User], rps: list[ResourceProvider]):
        generate_bids(users, rps)
        A = exec_online_pmmra(users, rps)
        for i in rps:
            for j in users:
                if A[i.idx][j.idx] == 1:
                    return Event(self.tick_secs, tick, j, i)
        return None

    def get_next_event(self):
        event = None

        if len(self.events) == 0:
            # edge case: no events yet
            event = self.get_event(0, self.all_users, self.all_rps)

        else:
            # work through the tick events to find the time of the next event
            event_ticks = self.get_event_ticks()
            for tick, events in event_ticks:
                # get the available users and rps at the given event time
                satisfied_users, busy_rps, available_users, available_rps = self.get_status(tick)
                
                # are there any available? then create an event
                if len(available_users) > 0 and len(available_rps) > 0:
                    event = self.get_event(tick, available_users, available_rps)
                    if event != None:
                        break
        
        if event != None:
            self.events.append(event)
        return event
    
    def generate_events(self):
        event = self.get_next_event()
        while event != None:
            event = self.get_next_event()

if __name__ == "__main__":
    all_users = gen_users(50, plot=False)
    all_rps = gen_resource_providers(all_users, 5, plot=False)
    auctioneer = Auctioneer(1, all_users, all_rps)
    auctioneer.generate_events()
    
    event_ticks = auctioneer.get_event_ticks()
    for tick, events in event_ticks:
        rp_indexes = [event.rp.idx for event in events]
        rp_indexes = np.unique(rp_indexes)
        print(f"{tick}: {rp_indexes}")
    _, _, available_users, _ = auctioneer.get_status(event_ticks[-1][0])
    print(f"{len(available_users)} unsatisfied users")