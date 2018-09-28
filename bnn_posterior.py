import copy
import time

import numpy as np
import matplotlib.pyplot as plt
import autograd
import autograd.misc.flatten as flatten
import autograd.numpy as ag_np

from new_prior import make_nn_params_as_dicts, make_nn_params_as_lists
from new_prior import flatten_dicts, flatten_lists, unflatten_vector
from activations import relu, tanh, rbf


x_train = np.asarray([-2.,    -1.8,   -1.,  1.,  1.8,     2.])
y_train = np.asarray([-3.,  0.2224,    3.,  3.,  0.2224, -3.])
sigma = 0.1


def U(q):
    """
    Returns the potential energy (float) of a given
    neural network configuration. Assumes x_train has been set already.

    Arguments:
    - q: A Numpy array corresponding to weights and biases of a NN
    """
    y_preds = ag_np.array([nn_predict(x=x, nn_param_list=q) for x in x_train])
    neg_log_L = ag_np.sum(ag_np.square(y_preds - y_train))/(2*sigma**2)
    neg_log_prior = ag_np.sum(ag_np.square(q))/2
    return neg_log_L + neg_log_prior


grad_U = autograd.grad(U) # Calculate the gradient of potential energy


def K(p):
    """
    Calculates the kinetic energy of some given momentum values and returns
    a float.

    Arguments:
    - p: Numpy array of momentum values
    """
    return ag_np.sum(ag_np.square(p)) / 2


def nn_predict(x, nn_param_list=None, act=ag_np.tanh):
    """
    This neural net prediction is terrible, and only works for the
    architecture with one hidden layer of 10 units. This was partly
    for desperation/debugging purposes, but had a neat side effect
    of making computation blazingly fast.

    Arguments: 
    - x: A scalar to predict on
    - nn_param_list: Numpy array of 31 NN parameters
    - act: Activation function to use (default=tanh)
    """
    h = act(ag_np.dot(x, nn_param_list[:10]) + nn_param_list[10:20])
    y = ag_np.dot(h,nn_param_list[20:30] + nn_param_list[30])
    return y


def run_HMC_sampler(init_bnn_params=None, n_hmc_iters=1000, n_leapfrog_steps=25,
                    eps=0.001, random_seed=42, U=U,
                    K=K, grad_U=grad_U):
    """ Run HMC sampler for many iterations (many proposals)

    Returns
    -------
    bnn_samples : list
        List of samples of NN parameters produced by HMC
        Can be viewed as 'approximate' posterior samples if chain runs to convergence.
    info : dict
        Tracks energy values at each iteration and other diagnostics.

    
    """
    # Create random-number-generator with specific seed for reproducibility
    prng = np.random.RandomState(int(random_seed))

    # Set initial bnn params
    cur_q = init_bnn_params
    cur_U = U(cur_q)

    bnn_samples = []
    energies = []
    energies.append(cur_U)
    start_time_sec = time.time()

    n_accept = 0
    for t in range(n_hmc_iters):

        cur_p = prng.normal(size=31)

        # Create PROPOSED configuration
        prop_q, prop_p = make_proposal_via_leapfrog_steps(
            cur_q, cur_p,
            n_leapfrog_steps=n_leapfrog_steps,
            eps=eps,
            grad_U=grad_U)
s
        prop_U = U(prop_q)
        cur_K = K(cur_p)
        prop_K = K(prop_p)
        accept_prob = ag_np.exp(cur_U-prop_U+cur_K-prop_K)
      

        # Draw random value from (0,1) to determine if we accept or not
        if prng.rand() < accept_prob:
            n_accept += 1
            cur_q, cur_U = (prop_q, prop_U)

        # Update list of samples from "posterior"

        bnn_samples.append(cur_q)
        energies.append(cur_U)

        # Print some diagnostics every 50 iters
        if t < 5 or ((t+1) % 50 == 0) or (t+1) == n_hmc_iters:
            accept_rate = float(n_accept) / float(t+1)
            print("iter %6d/%d after %7.1f sec | accept_rate %.6f" % (
                t+1, n_hmc_iters, time.time() - start_time_sec, accept_rate))

    return (
        bnn_samples,
        energies,
        dict(
            n_accept=n_accept,
            n_hmc_iters=n_hmc_iters,
            accept_rate=accept_rate),
        )


def make_proposal_via_leapfrog_steps(
        cur_bnn_params, cur_momentum_vec,
        n_leapfrog_steps=25,
        eps=0.001,
        grad_U=grad_U):
    """ Construct one HMC proposal via leapfrog integration

    Returns
    -------
    prop_bnn_params : same type/size as cur_bnn_params
    prop_momentum_vec : same type/size as cur_momentum_vec

    """
    # Initialize proposed variables as copies of current values
    q = copy.deepcopy(cur_bnn_params)
    p = copy.deepcopy(cur_momentum_vec)

    p = p - (eps * grad_U(q) / 2)

    for step_id in range(n_leapfrog_steps):
        q = q + (eps*p)
        if step_id < (n_leapfrog_steps - 1):
            p = p - (eps * grad_U(q))
        else:
            p = -1 * (p - (eps * grad_U(q) / 2))

    return q, p    

def plot_lines(x, lines, train=False):
    """
    Plots one figure with some x, and a bunch of y's from lines
    """
    plt.figure()
    if train:
        plt.plot(x_train, y_train, 'rx')
    for l in lines:
        plt.plot(x, l, '.-')


def multi_predict(x, bnn_configs):
    """
    Returns predictions on x from each of the BNN configrations passed in.
    The returned array is of size (N x S).

    Arguments:
    - x: A Numpy array of inputs of size S
    - bnn_configs: A Numpy array (N x 31)
    """
    multi_preds = []
    for bnn_config in bnn_configs:
        preds = ag_np.array([nn_predict(x=i, nn_param_list=bnn_config) for i in x])
        multi_preds.append(preds)
    return ag_np.array(multi_preds)


if __name__ == '__main__':
    """
    I apologize for not making this into a function. I again can only
    say that I was getting desperate.
    """
    x_grid = np.linspace(-20,20,200)
    chains = 1
    chain_length = 2000
    burnin = 1000
    samples = 10
    posteriors = []
    energies = []
    for chain in range(chains):
        nn_params = np.random.normal(size=31)
        post, es, info = run_HMC_sampler(n_hmc_iters=chain_length,
                                    init_bnn_params=nn_params)
        print(info)
        energies.append(es)
        posteriors.append(post)
        ind = np.random.choice(range(burnin, chain_length), samples, replace=False)
        post_samples = np.array(post)[ind]
        plot_lines(x_grid, (multi_predict(x_grid, post_samples)), train=True)
        plt.savefig('bnn_posterior_' + str(plt.gcf().number) + '.png',
                    # bbox_inches='tight')
    plot_lines(range(len(energies[0])), energies)
    plt.savefig('potential_energies.png', bbox_inches='tight')
    func_samples = multi_predict(x_grid, posteriors[0][1400:1900])
    mean_sample = np.mean(func_samples, axis=0)
    std = np.std(func_samples, axis=0)
    plt.figure()
    plt.plot(x_train, y_train, 'rx')
    plt.plot(x_grid, mean_sample, 'k-')
    plt.gca().fill_between(x_grid.flat, mean_sample-2*std, mean_sample+2*std,
                           color="#dddddd")
    plt.savefig('posterior_samples.png', bbox_inches='tight')
    plt.show()