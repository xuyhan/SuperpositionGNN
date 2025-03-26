import math

def sparcity_calculator(num_nodes, p, num_features):
    """
    Returns the probability that there is at least one pair of adjacent
    occupied sites in a lattice of n sites where each site is occupied
    independently with probability p.
    """
    # Compute the probability that there is no adjacent pair.
    prob_no_pair = 0.0
    # Modify definition of p
    p = p/num_features
    # The maximum number of occupied sites without adjacent ones is floor((n+1)/2)
    max_occupied = (num_nodes + 1) // 2
    for k in range(0, max_occupied + 1):
        # Number of ways to place k occupied sites with no adjacent ones:
        ways = math.comb(num_nodes - k + 1, k)
        prob_no_pair += ways * (p**k) * ((1-p)**(num_nodes-k))
    return prob_no_pair

print(sparcity_calculator(20, 0.8, 5))