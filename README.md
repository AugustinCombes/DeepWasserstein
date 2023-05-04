# Does Wasserstein-GAN approximate Wasserstein distances?

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AugustinCombes/DeepWasserstein/blob/main/main.ipynb)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/AugustinCombes/DeepWasserstein/main?labpath=main.ipynb)

*Group project for the [Optimal Transport course](http://marcocuturi.net/ot.html) at ENSAE teached by Marco Cuturi (spring 2023).*

- Augustin Combes
- Gabriel Watkinson

---

## The repository

The result of our experiments are presented in the [`main.ipynb`](https://github.com/AugustinCombes/DeepWasserstein/blob/main/main.ipynb) notebook. It is self contained and can be runned to replicate all of our experiments.

You can run the notebook interactively using [Google Colab](https://colab.research.google.com/github/AugustinCombes/DeepWasserstein/blob/main/main.ipynb) or [Binder](https://mybinder.org/v2/gh/AugustinCombes/DeepWasserstein/main?labpath=main.ipynb) by clicking on the badges above.

## Our experiments

### I. Motivations

The goal of this notebook is to explore whether Wasserstein-GAN (WGAN) can effectively approximate Wasserstein distances. WGAN, introduced in the 2017 [paper](https://arxiv.org/abs/1701.07875) by Arjovsky et al. [1], proposes a neural network-based proxy for the 1-Wasserstein distance, but it is unclear how well this approximation holds up in practice.

To investigate this question, we will implement the WGAN approach to solve Optimal Transport and compare it with other approaches, such as Sinkhorn divergence. Our aim is to determine if WGAN can compute a quantity that is truly similar to “true” optimal transport.

### II. Context

The WGAN paper proposes a new approach to learning a probability distribution by leveraging Optimal Transport theory.
Traditionally, learning a probability distribution involves maximizing the likelihood on the data across a family of parametric densities, denoted as $(P_\theta)_{\theta \in \mathbb{R}}$. This is equivalent to minimizing the Kullback-Leibler divergence between the real distribution $\mathbb{P}_r$ and the model distribution $\mathbb{P}_\theta$: $KL(\mathbb{P}_r|| \mathbb{P}_\theta)$. However, in many cases, the model density $P_\theta$ does not exist, and the Kullback-Leibler divergence is undefined.

To remedy to this problem, sampling directly from the target distribution $\mathbb{P}_\theta$ using a generator that maps random noise to a sample from the target distribution is another common approach, most known for its use in the [Variational Auto-Encoder](https://arxiv.org/abs/1312.6114) (VAE) [2] and the [original GAN](https://arxiv.org/abs/1406.2661) [3].

Arjovsky et al. propose a new approach based on Optimal Transport theory, expanding upon the idea of GANs. Traditional GANs are notoriously difficult to train, as a unstable equilibrium between the generator and discriminator is needed, and their results may suffer from mode collapse, in which the generator only produces a few samples that are very similar. WGAN is more stable and easier to tune than traditional GANs, using a proxy to the Wasserstein distance via neural nets. Our aim in this project is to investigate whether WGAN is a promising approach to solving Optimal Transport problems.


> [1]: Arjovsky, Martin, Soumith Chintala, and Léon Bottou. "Wasserstein generative adversarial networks." International conference on machine learning. PMLR, 2017. https://arxiv.org/abs/1701.07875  
> [2]: Kingma, Diederik P., and Max Welling. "Auto-encoding variational bayes." arXiv preprint arXiv:1312.6114 (2013). https://arxiv.org/abs/1312.6114  
> [3]: Goodfellow, Ian, et al. "Generative adversarial networks." Communications of the ACM 63.11 (2020): 139-144. https://arxiv.org/abs/1406.2661

### III. Theory

Unlike in GANs, where the generator's loss is the binary cross-entropy between the discriminator's output and a target value indicating whether the generated sample is real or fake, WGAN use the Wasserstein-1 distance (also known as the Earth Mover's Distance) to measure the difference between the real and generated distributions:
```math
W(\mathbb{P}_r, \mathbb{P}_g) = \inf_{\gamma \in \Pi(\mathbb{P}_r, \mathbb{P}_g)}\mathbb{E}_{(x, y)\sim\gamma}[\vert\vert x-y\vert\vert]
```
where $\Pi(\mathbb{P}_r, \mathbb{P}_g)$ denotes all joint distributions $\gamma (x,y)$ whose marginals are respectively $\mathbb{P}_r$ and $\mathbb{P}_g$.

However, this formulation is highly impractical, as it is not tractable and can't be used in practice. Instead, WGAN uses the Kantorovich-Rubinstein equivalent:
```math
W(\mathbb{P}_r, \mathbb{P}_\theta) = \sup_{\vert\vert f_L\vert \vert \leq 1}\mathbb{E}_{x_1\sim\mathbb{P}_r}[f(x)] - \mathbb{E}_{x_2\sim\mathbb{P}_\theta}[f(x_2)]
```

More precisely, we will search $f$ as a parametrized function $f_w$, for $w\in\mathcal{W}$, being $K$-Lipschitz for some $K$. If the supremum is attained for some $w$, then we have a calculation of $W(\mathbb{P}_r, \mathbb{P}_\theta)$ up to a constant factor.

Therefore, to approximate the Wasserstein distance, we will train a neural network to maximize the following objective function:
```math
\max_{w\in\mathcal{W}} \mathbb{E}_{x_1\sim\mathbb{P}_1}[f_w(x_1)] - \mathbb{E}_{x_2\sim\mathbb{P}_2}[f_w(x_2)]
```

The resulting value should be close to the "true" Wasserstein distance between the two distributions.

### IV. Numerical estimation

In practice, denoting with $\mathbb{\hat E}$ the empirical mean operator, we will optimize numerically the empirical counterpart of this optimization program, that is:
```math
\max_{w\in\mathcal{W}} \mathbb{\hat E}_{x_1\sim\mathbb{P}_1}[f_w(x_1)] - \mathbb{\hat E}_{x_2\sim\mathbb{P}_2}[f_w(x_2)]
```

We will estimate the function $f_w$ as an MLP.

Enforcing the 1-Lipschitz constraint can be done using various techniques:

#### IV.1. Clipping

As proposed in the original WGAN paper, clipping the weights of the network is a simple way to ensure the constraint.

Denoting $l_{w_1}, \dots, l_{w_q}$ as the successive layers of the MLP, such that $f_w = l_{w_1} \circ \dots \circ l_{w_q}$, and using ReLU (1-Lipschitz) activation, a sufficient way of ensuring 1-Lipschitz $f_w$ is to ensure each layer to be 1-Lispchitz.

Then, a sufficient way of ensuring each linear layer $i$ to be 1-Lipschitz is to ensure $\vert\vert w_i\vert\vert_{\infty} \leq 1$. That is, clipping the weights to $[-1, 1]$ ensures the resulting estimate $f_w$ to be 1-Lipschitz.

#### IV.2. Penalization

As proposed in other follow-up papers, another way to ensure the constraint is to penalize when the parameters do not respect it directly in the loss.

That is, we then optimize on the following penalized optimization program:
```math
\max_{w\in\mathcal{W}} \mathbb{\hat E}_{x_1\sim\mathbb{P}_1}[f_w(x_1)] - \mathbb{\hat E}_{x_2\sim\mathbb{P}_2}[f_w(x_2)] + \lambda\times \vert\vert w - 1(w)\vert\vert _2^2
```

where $\lambda$ is an additional hyperparameter that controls the amount of penalization given to the constraint, and $1(w)$ is an object of same size than $w$ constant to $1$.

We explore both of these two methods in this notebook.

## Running the notebook locally

If you want to run the notebook locally, feel free to use poetry to create and install the environment. You can do so by running the following commands:

```bash
git clone https://github.com/AugustinCombes/DeepWasserstein.git
cd DeepWasserstein
```

```bash
# curl -sSL https://install.python-poetry.org | python3 -  # if you need to install poetry, see https://python-poetry.org/docs/ for details
poetry install        # to create the environment from the poetry.lock file
# poetry shell        # to spawn a shell in the environment
# pre-commit install  # if you want to use pre-commit hooks
# poe update_jax_12   # to update jaxlib using cuda 12
```

Else, you can use the `requirements.txt` file to install the dependencies with pip:

```bash
pip install -r requirements.txt
```
