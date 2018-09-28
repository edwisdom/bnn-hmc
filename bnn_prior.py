import itertools
import numpy as np
import autograd.numpy as ag_np
import autograd
import matplotlib.pyplot as plt
from activations import relu, tanh, rbf



def sample_gaussian_prior(size):
    """Returns size samples from a Gaussian of mean=0 and std=1"""
    return np.random.normal(size=size)

def make_nn_params_as_dicts(
        n_hiddens=[10],
        n_dims_input=1,
        n_dims_output=1,
        weight_fill_func=sample_gaussian_prior,
        bias_fill_func=sample_gaussian_prior):
    nn_param_list = []
    n_hiddens = [n_dims_input] + n_hiddens + [n_dims_output]

    # Given full network size list is [a, b, c, d, e]
    # For loop should loop over (a,b) , (b,c) , (c,d) , (d,e)
    for n_in, n_out in zip(n_hiddens[:-1], n_hiddens[1:]):
        nn_param_list.append(
            dict(
                w=weight_fill_func((n_in, n_out)),
                b=bias_fill_func((n_out,)),
            ))
    return nn_param_list


def pretty_print_nn_param_list(nn_param_list_of_dict):
    """ Create pretty display of the parameters at each layer
    """
    for ll, layer_dict in enumerate(nn_param_list_of_dict):
        print("Layer %d" % ll)
        print("  w | size %9s | %s" % (layer_dict['w'].shape, layer_dict['w'].flatten()))
        print("  b | size %9s | %s" % (layer_dict['b'].shape, layer_dict['b'].flatten()))


def make_nn_params_as_lists(
        n_hiddens=[10],
        n_dims_input=1,
        n_dims_output=1,
        weight_fill_func=sample_gaussian_prior,
        bias_fill_func=sample_gaussian_prior):
    """Simply returns params as a list of lists instead"""
    nn_dicts = make_nn_params_as_dicts(n_hiddens=n_hiddens,
                                      n_dims_input=n_dims_input,
                                      n_dims_output=n_dims_output,
                                      weight_fill_func=weight_fill_func,
                                      bias_fill_func=bias_fill_func)
    return flatten_dicts(nn_dicts)


def flatten_dicts(nn_params_dicts):
    """Flattens a list of dictionaries into a list of lists"""
    return [list(nnp_d.values()) for nnp_d in nn_params_dicts]    


def flatten_lists(nn_params_lists):
    """Flattens a list of lists into one big list"""
    flattened = np.array([])
    for layer in nn_params_lists:
        for p_list in layer:
            flattened = ag_np.concatenate((flattened, ag_np.ravel(p_list)))
    return flattened

def unflatten_vector(vector, n_hiddens=[10], n_dims_input=1, n_dims_output=1):
    """Unflattens a vector into its list-of-lists representations for a NN"""
    unflat_list = []
    start = 0
    layers = n_hiddens + [n_dims_output]
    for h_i in range(len(layers)): 
        if h_i == 0:
            input_size = n_dims_input
        else:
            input_size = layers[h_i-1]
        biases = layers[h_i]
        weights_i = start + input_size*biases
        unflat_list.append([ag_np.array(vector[start:weights_i]).reshape((input_size, biases)), 
                            ag_np.array(vector[weights_i:weights_i+biases])])
        start = weights_i + biases
    return unflat_list


def nn_predict(x=None, nn_param_list=None, act=tanh):
    """
    Makes a prediction on x given some neural network parameters as
    a list of dicts or list of lists
    """
    j, k = ('w','b') if type(nn_param_list[0]) == dict else (0,1)
    for layer_id, layer_array in enumerate(nn_param_list):
        if layer_id == 0:
            if x.ndim > 1:
                in_arr = x
            else:
                if x.size == nn_param_list[0][j].shape[0]:
                    in_arr = x[ag_np.newaxis,:]
                else:
                    in_arr = x[:,ag_np.newaxis]
        else:
            in_arr = act(out_arr)
        out_arr = ag_np.dot(in_arr, layer_array[j]) + layer_array[k]
    return ag_np.squeeze(out_arr)


def sample_bnn_prior(x, architectures, activations, samples=5):
    """
    Returns multiple samples from all BNN priors that are combinations
    of architectures and activations passed in.

    Arguments:
    - x: Numpy array of points to evaluate the prior on
    - architectures: List of architectures, where each architecture is
    a list of hidden layer sizes
    - activations: List of activation functions
    - samples: Number of samples for each prior (default=5)
    """
    all_bnns = []
    for a_a in list(itertools.product(architectures, activations)):
        all_samples = []
        for s in range(samples):
            nn_params = make_nn_params_as_lists(n_hiddens=a_a[0])
            preds = nn_predict(x, nn_param_list=nn_params, act=a_a[1])
            all_samples.append(preds)
        all_bnns.append(all_samples)
    return all_bnns


def plot_samples(results):
    """Plots samples from multiple BNN priors and saves the plots"""
    for bnn in results:
        plt.figure()
        for sample in bnn:
            plt.plot(x_grid, sample, '.-')
        plt.savefig('bnn_prior_' + str(plt.gcf().number) + str('.png'), 
                    bbox_inches='tight')
    plt.show() 


def test_unflat():
    """
    Unit test for the flattening/unflattening of the neural net
    """
    nn_params = make_nn_params_as_lists()
    print("The following two shape-lists should be the same")
    print([p.shape for p_list in nn_params for p in p_list])
    preds = nn_predict(x=x_grid, nn_param_list=nn_params)
    flat = flatten_lists(nn_params)
    unflat = unflatten_vector(flat)
    print([p.shape for p_list in unflat for p in p_list])
    u_preds = nn_predict(x=x_grid, nn_param_list=unflat)
    print("This array must hold all trues")
    print(preds == u_preds)


if __name__ == '__main__':
    x_grid = np.linspace(-20, 20, 200)
    ys = sample_bnn_prior(x=x_grid, 
                          architectures=[[2], [10], [2,2], [10,10]],
                          activations=[relu, tanh, rbf])
    plot_samples(ys)
    # print(test_unflat())