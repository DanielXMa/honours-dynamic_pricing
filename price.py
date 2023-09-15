import numpy as np

def price_fixed_price(pbase, psurge, supply, demand): # fixed pbase and psurge
    p_prime = np.zeros(len(supply))
    for i in range(len(supply)):
        if demand[i] > supply[i]:
            p_prime[i] = pbase + np.max(psurge)*((demand[i] - supply[i])/demand[i])
        else:
            p_prime[i] = pbase
    return p_prime

def price_fixed_supply(pbase, psurge, supply, demand): # fixed supply
    p_prime = np.zeros(len(pbase))
    for i in range(len(demand)):
        if demand[i] > supply:
            p_prime[i] = pbase[i] + np.max(psurge)*((demand[i] - supply)/demand[i])
        else:
            p_prime[i] = pbase[i]
    return p_prime

def price_fixed_demand(pbase, psurge, supply, demand): # fixed demand
    p_prime = np.zeros(len(pbase))
    for i in range(len(supply)):
        if demand > supply[i]:
            p_prime[i] = pbase[i] + np.max(psurge)*((demand - supply[i])/demand)
        else:
            p_prime[i] = pbase[i]
    return p_prime
