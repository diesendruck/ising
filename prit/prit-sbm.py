# coding: utf-8

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from datetime import datetime
import math

def main(verbose=False):
    # Define 'global' variables for this test.
    n = 50
    alpha = 0.5
    num_clusters = 5

    # Run whole process (sample stack and produce the A-matrix) many times.
    plot_list = []
    for i in range(16):
        a = run_process(n, alpha, num_clusters)
        plot_list.append(a)

    # Visualize sampled A-matrices.
    path = "/Users/mauricediesendruck/Google Drive/0-LIZHEN RESEARCH/prit"
    title = ((r'$A$-Matrix: '+r'$n={}$, '+r'$k={}$, '+
              r'$\alpha={}$').format(n, num_clusters, alpha))

    visualize_plotlist(plot_list, title, path)

def run_process(n, alpha, num_clusters):
    """ Performs full hierarchical sampling to produce the A-matrix."""
    # Set Dirichlet hyperparams and sample pi's.
    alpha_vec = np.empty(num_clusters); alpha_vec.fill(alpha)
    pi = np.random.dirichlet(alpha_vec, 1)[0]
    print "Vector of cluster probabilities:{}".format(np.around(pi, decimals=2))

    # Sample vector z of cluster assignments: Multinomial distr with pi weights.
    z = np.empty(n); z.fill(None)
    for i in xrange(n):
        z[i]=np.random.choice(a=num_clusters, size=1, p=pi)

    # Sample Q matrix (cluster link probabilities).
    q = np.array([[np.random.uniform(0,1,1)[0] for j in range(num_clusters)] for
                  i in range(num_clusters)])

    # Sample A, given z and Q.
    a = sample_a(z, q)

    return a

def sample_a(z, q):
    """ Sample A-matrix from assignments and cluster link probabilities."""
    n = len(z)
    # Define function to fetch link probability based on node cluster.
    def c2c_link_prob(i, j):
        i_cluster = z[i]
        j_cluster = z[j]
        prob = q[i_cluster, j_cluster]
        return prob

    # Sample each link with Bernoulli p, based on cluster membership.
    a = np.array([[rbern(c2c_link_prob(i, j)) for j in range(n)] for i in
                  range(n)])
    # Sort column order by cluster, then sort row order by cluster.
    a = a[:, np.argsort(z)][np.argsort(z), :]

    return a

def visualize_plotlist(plot_list, title, path):
    """Plot list of adjacency matrics."""
    # Choose figure dimensions to be square, and big enough for plot_list.
    l = len(plot_list)
    fig_dim = math.sqrt(l)
    if int(fig_dim)**2 < l:
        nrows = int(fig_dim)
        ncols = int(math.ceil(l/nrows))
    else:
        nrows = ncols = int(fig_dim)
    # Set up figure with multiple subplots.
    fig, axes = plt.subplots(nrows, ncols, figsize=(8,8))
    axes = axes.ravel()
    plt.suptitle(title)

    # Create subplots with several trials per unique theta.
    for i in xrange(len(plot_list)):
        axes[i].imshow(plot_list[i], interpolation='none', cmap='GnBu')
        axes[i].tick_params(labelsize=6)

    # Save figures to directory.
    os.chdir(path)
    plt.savefig('fig-'+time.strftime('%Y%m%d_%H:%M:%S')+'.png', format='png',
                dpi=1200)

def rbern(p):
    r = np.random.binomial(1, p)
    return r


start_time = datetime.now()
a = main(verbose=False)
print datetime.now() - start_time


