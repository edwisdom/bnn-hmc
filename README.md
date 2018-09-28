# Bayesian Neural Nets and Hamiltonian Monte Carlo

## Getting Started

These instructions will allow you to run this project on your local machine.

### Install Requirements

Once you have a virtual environment in Python, you can simply install necessary packages with: `pip install -r requirements.txt`

### Clone This Repository

```
git clone https://github.com/edwisdom/bnn-hmc
```

### Run Models

Run the Bayesian neural net prior with:

```
python bnn_prior.py
```

Run the Bayesian neural net posterior (and Hamiltonian Monte Carlo) with:

```
python bnn_posterior.py
```


## Sampling from a Bayesian Neural Net Prior


<img align="left" width="275" height="300" src="https://github.com/edwisdom/bnn-hmc/blob/master/bnn_prior_1.png">
<img align="left" width="275" height="300" src="https://github.com/edwisdom/bnn-hmc/blob/master/bnn_prior_2.png">
<img align="left" width="275" height="300" src="https://github.com/edwisdom/bnn-hmc/blob/master/bnn_prior_3.png">
Figure 1: 5 samples from a BNN prior (hidden layer=[2]) with relu, tanh, and rbf activations 
<img align="left" width="275" height="300" src="https://github.com/edwisdom/bnn-hmc/blob/master/bnn_prior_4.png">
<img align="left" width="275" height="300" src="https://github.com/edwisdom/bnn-hmc/blob/master/bnn_prior_5.png">
<img align="left" width="275" height="300" src="https://github.com/edwisdom/bnn-hmc/blob/master/bnn_prior_6.png">
Figure 2: 5 samples from a BNN prior (hidden layer=[10]) with relu, tanh, and rbf activations
<img align="left" width="275" height="300" src="https://github.com/edwisdom/bnn-hmc/blob/master/bnn_prior_7.png">
<img align="left" width="275" height="300" src="https://github.com/edwisdom/bnn-hmc/blob/master/bnn_prior_8.png">
<img align="left" width="275" height="300" src="https://github.com/edwisdom/bnn-hmc/blob/master/bnn_prior_9.png">
Figure 3: 5 samples from a BNN prior (hidden layer=[2, 2]) with relu, tanh, and rbf activations 
<img align="left" width="275" height="300" src="https://github.com/edwisdom/bnn-hmc/blob/master/bnn_prior_10.png">
<img align="left" width="275" height="300" src="https://github.com/edwisdom/bnn-hmc/blob/master/bnn_prior_11.png">
<img align="left" width="275" height="300" src="https://github.com/edwisdom/bnn-hmc/blob/master/bnn_prior_12.png">
Figure 4: 5 samples from a BNN prior (hidden layer=[10,10]) with relu, tanh, and rbf activations


### Qualitative Trends in the Prior

The samples from the prior, as Figure 1 shows, clearly depend heavily on the choice of activation functions. Whereas the relu prior samples are essentially two piecewise lines, the tanh prior samples look like sigmoid curves (which reflects the underlying activation functions). 

Although these activation functions are simple, even with just one hidden layer of 2 units, different priors have a large range at each possible input value (except the relu activation at x=0, predictably so). This suggests that our priors are fairly flexible, and can fit a lot of different training data. Another clear trend is that with more hidden layers, individual prior samples become more complex with more local extrema. 

## Sampling from a Bayesian Neural Net Posterior with Hamiltonian Monte Carlo

### Convergence

<img align="left" width="600" height="600" src="https://github.com/edwisdom/bnn-hmc/blob/master/potential_energies.png">

Figure 5: Potential energies over Hamiltonian Monte Carlo iterations for 3 different chains

As we see in Figure 5, each of the chains of Hamiltonian Monte Carlo rapidly converge to low potential energy values, which means that the samples we're getting are from high probability-mass regions of the posterior. This is the primary reason why Hamiltonian Monte Carlo is preferred over Metropolis-style (MCMC) methods, since the latter is unlikely to explore a wide space while still remaining in high-probability regions. 

To see an interactive demo of this principle, [see here](https://chi-feng.github.io/mcmc-demo/app.html#HamiltonianMC,banana).

<br />
<br />
<br />

### Posterior Shapes

<img align="left" width="275" height="300" src="https://github.com/edwisdom/bnn-hmc/blob/master/bnn_posterior_1.png">
<img align="left" width="275" height="300" src="https://github.com/edwisdom/bnn-hmc/blob/master/bnn_posterior_1.png">
<img align="left" width="275" height="300" src="https://github.com/edwisdom/bnn-hmc/blob/master/bnn_posterior_1.png">

Figure 6: 10 samples from 3 different BNN posteriors sampled using HMC with epsilon=0.001 and L=25


As we can see in Figure 6, each of the posterior samples wraps tightly around the training data. However, the predictions that are significantly further away from any training data are much more variable. 


### Uncertainty and the Deficiency of Point Estimates

<img align="left" width="600" height="600" src="https://github.com/edwisdom/bnn-hmc/blob/master/posterior_samples.png">

Figure 7: 500 posterior function samples from a single BNN posterior trained with epsilon=0.001 and L=25

Figure 7 shows how our posterior has much greater uncertainty at input points that are far away from its training data. Its estimates of these values are largely dominated by the prior. This kind of model gives us an edge over traditional point-estimate neural networks because they give a distribution over our parameters and allow us to quantify our certainty about predictions. These models have the potential to be both more interpretable and more capable of [detecting adversarial perturbations](https://arxiv.org/abs/1711.08244).


## Future Work

In the future, I would like to explore the following:

1. Tuning hyperparameters epsilon and L more exhaustively and systematically
2. Applying this model to real-world data and comparing it to neural networks that take similar time to train
3. Implementing the [NUTS](https://arxiv.org/abs/1111.4246) (No U-Turn Sampler), which is currently the best known Monte Carlo sampling technique for Bayesian neural nets

## Credits

A huge thanks to Prof. Michael Hughes, who supervised this work, and Daniel Dinjian and Julie Jiang for thinking through the technical nitty-gritty with me.