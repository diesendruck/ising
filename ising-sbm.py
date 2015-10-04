# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import os
import time

def main(verbose=False):
    var_set = [[30, 0.9, 0.5, 0.01, 15.0, 0.5],
               [30, 0.9, 0.5, 0.01, 5.0, 0.5],
               [30, 0.9, 0.5, 0.01, 1.0, 0.5],
               [30, 0.9, 0.5, 0.01, -1.0, 0.5],
               [30, 0.9, 0.5, 0.01, -5.0, 0.5],
               [30, 0.9, 0.5, 0.01, -15.0, 0.5]]
    plot_list = []

    # Use same adjacency matrix for all iterations of other variables.
    n = 30
    adj_matrix_intensity = 0.5
    adj = sample_adj_matrix(n, adj_matrix_intensity)

    for v in var_set:
        n, p_pos, p_neg, p_btwn, theta_fill_value, adj_matrix_intensity = (
            var_set[0])
        a = sample_a(n, p_pos, p_neg, p_btwn, theta_fill_value, adj, verbose)
        plot_list.append(a)

    visualize(plot_list)
    return a

def visualize(plot_list):
    # Plot test figures.
    fig = plt.figure()
    f1 = fig.add_subplot(231)
    f1.imshow(plot_list[0], interpolation='none', cmap='GnBu')
    f1.set_title('theta=15.0')

    f2 = fig.add_subplot(232)
    f2.imshow(plot_list[1], interpolation='none', cmap='GnBu')
    f2.set_title('theta=5.0')

    f3 = fig.add_subplot(233)
    f3.imshow(plot_list[2], interpolation='none', cmap='GnBu')
    f3.set_title('theta=1.0')

    f4 = fig.add_subplot(234)
    f4.imshow(plot_list[3], interpolation='none', cmap='GnBu')
    f4.set_title('theta=-1.0')

    f5 = fig.add_subplot(235)
    f5.imshow(plot_list[4], interpolation='none', cmap='GnBu')
    f5.set_title('theta=-5.0')

    f6 = fig.add_subplot(236)
    f6.imshow(plot_list[5], interpolation='none', cmap='GnBu')
    f6.set_title('theta=-15.0')

    # Save figures to directory.
    path = '/Users/mauricediesendruck/Google Drive/0-LIZHEN RESEARCH/ising/'
    os.chdir(path)
    plt.savefig('fig-'+time.strftime('%Y%m%d_%H:%M:%S'))

def sample_a(n, p_pos, p_neg, p_btwn, theta_fill_value, adj, verbose):
        theta = np.empty([n, n]); theta.fill(theta_fill_value)
        #adj = sample_adj_matrix(n, adj_matrix_intensity)
        z = sample_ising(theta, adj)
        q = build_q_matrix(z, p_pos, p_neg, p_btwn)
        a = sample_sbm(q, n)

        if verbose==True:
            summarize(n, p_pos, p_neg, p_btwn, theta_fill_value,
                      adj_matrix_intensity, adj, z, q, a)
        return a

def summarize(n, p_pos, p_neg, p_btwn, theta_fill_value, adj_matrix_intensity,
              adj, z, q, a):
    print('N: ', n)
    print('Pr(1): ', p_pos)
    print('Pr(-1): ', p_neg)
    print('Pr(between): ', p_btwn)
    print('Theta fill value: ', theta_fill_value)
    print('Adjacency matrix intensity: ', adj_matrix_intensity)
    print
    print(adj)
    print("Z vector: ")
    print(z)
    print("q matrix:")
    print(q)
    print("For q: ", check_symmetry(q))
    print("a matrix:")
    print(a)
    print("For a: ", check_symmetry(a))

def sample_adj_matrix(n, p):
    """Builds random adjacency matrix.

    Creates nxn adjacency matrix (1s and 0s) representing edges between nodes.
    Each edge is sampled as an independent Bernoulli random variable with
    probability p.

    Args:
        n: Number of nodes, and size of matrix adjacency matrix.
        p: Bernoulli probabiity for each edge.

    Returns:
        adj: Adjacency matrix.
    """
    adj = np.asarray([[rbern(p) for j in range(n)] for i in range(n)])
    adj = sym_matrix(adj)
    np.fill_diagonal(adj, 0)
    return adj

def build_q_matrix(z, p_pos, p_neg, p_btwn):
    """Builds q matrix from stochastic block model.

    Compares each element in z to every other element in z, assigning link
    probabilities according to the agreement between pairs of elements.

    Args:
        z: Vector of ising assignments.
        p_pos: Link probability for pair of elements in cluster +1.
        p_neg: Link probability for pair of elements in cluster -1.
        p_btwn: Link probability for pair of elements in opposite clusters.

    Returns:
        q: Q matrix of pairwise link probabilities, given the stochastic block
            model.
    """

    def cond(i, j):
        """Determines which probability value applies for a given pair.

        Args:
            i: Reference index of z vector.
            j: Comparison index of z vector.

        Returns:
            p: Probability value.
        """
        # Probability to return, which gets reassigned given conditions.
        p = 0

        # A point and itself gets a zero, so q has zeros on diagonal.
        if i == j:
            p = 0
        else:
            # When reference element is 1, return within-cluster-1 probability
            # or cross prob.
            if z[i] == 1:
                if z[i] == z[j]:  # if pair is equal, give cluster 1 probabiity
                    p = p_pos
                else:
                    p = p_btwn
            # When reference element is -1, return within-cluster-(-1)
            # probability or cross prob.
            elif z[i] == -1:
                if z[i] == z[j]:
                    p = p_neg
                else:
                    p = p_btwn
            else:
                p = "z[i] not in [1, -1]"
        return p

    n = len(z)
    # Evaluate over all z indices; here, indices are the range 0 to n-1.
    q = np.asarray([[cond(i, j) for j in range(n)] for i in range(n)])
    return q

def check_symmetry(q): return("Symmetry: ", (q.transpose() == q).all())

def sample_sbm(q, n):
    """Samples from the Stochastic Block Model (SBM) link probability matrix.

    Args:
        q: The link probability matrix.
        n: The number of rows (and equivalently, columns) of the matrix q.

    Returns:
        a: An instance of the link matrix, based on SBM probability matrix.
    """
    a = np.asarray([[rbern(q[i, j]) for j in range(n)] for i in range(n)])
    a = sym_matrix(a)
    return a

def rbern(p):
    r = np.random.binomial(1, p)
    return r

def sym_matrix(matrix, part="upper"):
    """Makes square, symmetric matrix, from matrix and upper/lower flag.

    Requires: import numpy as np

    Supply a square matrix and a flag like "upper" or "lower", and copy the
    chosen matrix part, symmetrically, to the other part. Diagonals are left
    alone. For example:
    matrix <- [[8, 1, 2],
               [0, 8, 4],
               [0, 0, 8]]
    sym_matrix(matrix, "upper") -> [[8, 1, 2],
                                    [1, 8, 4],
                                    [2, 4, 8]]

    Args:
        matrix: Square matrix.
        part: String indicating "upper" or "lower".

    Returns:
        m: Symmetric matrix, with either upper or lower copied across the
            diagonal.
    """
    n = matrix.shape[0]
    upper_indices = np.triu_indices(n, k=1)
    lower_indices = upper_indices[1], upper_indices[0]
    m = np.copy(matrix)
    if part=="upper":
        m[lower_indices] = m[upper_indices]
    elif part=="lower":
        m[upper_indices] = m[lower_indices]
    else:
        print("Give a good 'part' definition, e.g. 'upper' or 'lower'.")

    return m

def sample_ising(theta, adj):
    """Given a matrix of agreement parameters, samples binary ising vector.

    Samples vector of 1's and -1's from a Gibbs sampled Ising Distribution.

    Args:
        theta: Agreement parameter matrix; one agreement coefficient for each
            pair of nodes.
        adj: Adjacency matrix of pairwise edge connections (binary).

    Returns:
        z_sample: Vector of n values, each either 1 or -1.
    """
    # Set up parameters and variable storage.
    n = len(theta)  # Number of nodes in graph.
    num_trials = 500  # Number of times to run the Gibbs sampler.
    burn_in = 100  # Number of Gibbs samples to discard; must be < num_trials.
    z_chain = np.zeros([num_trials, n])  # Storage for Gibbs samples, by row.

    # Initialize and store first configuration of z's.
    z0 = np.random.choice([-1, 1], n)  # Initialize z's.
    z_chain[0,:] = z0  # Store initial values as first row of z_chain.

    # Run Gibbs.
    for t in range(1, num_trials):
        z = z_chain[t-1,:]
        for i in range(n):
            # Sample each z from its full Ising model conditional.
            # pi(z_i|z_not_i) = (1/C)*exp(sum(theta*z_i*z_j)), for j's with
            #     edges to i.
            # Use adjacency matrix as indicator to pick j's.
            # Evaluate for z_i=-1 and z_i=1, normalize, then sample.
            summation_terms_neg1 = [-adj[i, j]*theta[i, j]*z[j]
                if j>i else 0 for j in range(n)]
            summation_terms_pos1 = [+adj[i, j]*theta[i, j]*z[j]
                if j>i else 0 for j in range(n)]
            pn = unnorm_prob_neg1 = np.exp(sum(summation_terms_neg1))
            pp = unnorm_prob_pos1 = np.exp(sum(summation_terms_pos1))
            # Normalize probabilities.
            pr_neg1 = pn/(pn+pp)
            pr_pos1 = pp/(pn+pp)
            # Sample z_i
            z_i_value = np.random.choice([-1, 1], p=[pr_neg1, pr_pos1])
            # Store z_i value in z_chain.
            z_chain[t, i] = z_i_value

    # Sample a z from the z_chain.
    sample_index = np.random.randint(burn_in, len(z_chain))
    z_sample = z_chain[sample_index,:]

    return z_sample



a = main(verbose=False)


