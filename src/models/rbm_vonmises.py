# SPDX-License-Identifier: MIT
# src/models/rbm_vonmises.py

"""Module that defines the RBM_vonmises class."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D


class RBM_vonmises(nn.Module):
    """RBM with VonMises visible units and binary hidden units.

    This class implements an RBM which models the visible units as a random
    variable sampled from a von-Mises distribution. It learns data
    distributions using Contrastive Divergence (CD) or Persistent Contrastive
    Divergence (PCD) with Maximum likelihood estimation (MLE). It supports
    both Gibbs sampling and Langevin dynamics for MCMC.

    Notes:
        - Parameters: A, B, c
        - Contains a function to update parameters for just one batch.
        - Calculates data visible energy and model visible energy.
        - Difference between data and model visible energies is recorded.
        - One-step reconstruction mse is used as a training metric.

    Attributes:
        n_visible: Number of visisble units.
        n_hidden: Number of hidden units.
        A: Weight matrix between visible and hidden units.
        B_bias: Bias vector for visible units.
        h_bias: Bias vector for visible units.
        vA: Velocity/Momentum vector for Weight matrix updates when using momentum.
        vB: Velocity/Momentum vector for visible unit bias updates when using momentum
        vh_bias: Velocity/Momentum vector for hidden unit bias updates when using momentum
        persistent_v: Persistent visible-state used for persistent
            contrastive divergence (PCD).

    **Reference**:
        Kai Zhang and Sora Sakai,
        *Restricted Boltzmann Machines in Physics: Concepts, Theories, and Applications*.
        Throughout this module, this work is referred to as "the paper".
    """

    def __init__(self, n_visible: int, n_hidden: int):
        """Initiate the RBM_vonmises class.

        Args:
            n_visible: Number of visisble units.
            n_hidden: Number of hidden units.
        """
        super(RBM_vonmises, self).__init__()
        self.n_visible = n_visible  # nv
        self.n_hidden = n_hidden  # nh

        # Model parameters
        limit = 4.0 * math.sqrt(6.0 / (n_hidden + n_visible))

        self.A = nn.Parameter(
            torch.empty(n_hidden, n_visible).uniform_(-limit, limit)
        )  # (nh, nv)
        self.B = nn.Parameter(
            torch.empty(n_hidden, n_visible).uniform_(-limit, limit)
        )  # (nh, nv)
        self.h_bias = nn.Parameter(torch.zeros(n_hidden))  # (nh, )

        # Define momentums
        self.register_buffer("vA", torch.zeros_like(self.A))
        self.register_buffer("vB", torch.zeros_like(self.B))
        self.register_buffer("vh_bias", torch.zeros_like(self.h_bias))

        # Initialize persistent chain
        self.persistent_v = None

    def xi(self, v: torch.Tensor) -> torch.Tensor:
        """Compute the hidden pre-activation vector from the visible units.

        Note the implementation is equivalent to the formulation in the paper, but
        uses batched row vectors instead of individual column vectors.

        **Equations**:
            Mathematical (Paper, Section 3.6):
                ξ(v) = Acos(v) + Bsin(v)

            Implementation:
                ξ(v) = cos(v)Aᵀ + sin(v)Bᵀ + c

        **Shapes**:
            Mathematical (Paper):
                v: ``[nv]``
                A: ``[nh, nv]``
                B: ``[nh, nv]``
                ξ(v): ``[nh]``

            Implementation:
                Input (v): ``[batch_size, nv]``
                A: ``[nh, nv]``
                B: ``[nh, nv]``
                c: ``[nh]``
                Output (ξ(v)): ``[batch_size, nh]``

        Args:
            v: Batch of visible layer state vectors.

        Returns:
            ξ(v): Batch of hidden pre-activation vectors.
        """
        cosx = torch.cos(v)  # size [batch_size, nv]
        sinx = torch.sin(v)  # size [batch_size, nv]

        return (
            cosx @ self.A.T + sinx @ self.B.T + self.h_bias
        )  # size [batch_size, nh]

    def hW(self, h: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
        """Compute the visible pre-activation vectors from the hidden units.

        Note the implementation is equivalent to the formulation in the paper, but
        uses batched row vectors instead of individual column vectors.

        **Equations**:
            Mathematical (Paper, Section 3.6):
                αᵀ(h) = hᵀA
                βᵀ(h) = hᵀB

            Implementation:
                α(h) = h @ A
                β(h) = h @ B

        **Shapes**:
            Mathematical (Paper):
                h: ``[nh]``
                A: ``[nh, nv]``
                B: ``[nh, nv]``
                α(h): ``[nv]``
                β(h): ``[nv]``

            Implementation:
                Input (h): ``[batch_size, nh]``
                A: ``[nh, nv]``
                B: ``[nh, nv]``
                Output (α(h) or β(h)): ``[batch_size, nv]``

        Args:
            h: Batch of hidden layer state vectors.
            W: The hidden weight matrix, insert either the A or B
                matrix.

        Returns:
            α(h) or β(h): Batch of visible pre-activation vectors.
        """
        return h @ W

    def bernoulli_sampling(self, p: torch.Tensor) -> torch.Tensor:
        """Sampling from a Bernoulli distribution with prob p."""
        return torch.bernoulli(p)

    def v_to_h(self, v: torch.Tensor) -> torch.Tensor:
        """Sample a batch of hidden states from a batch of visible states.

        Computes probabilities using the visible units and then samples
        hidden units from a Bernoulli distribution.

        **Equations**:
            Mathematical (Paper, Section 2.2):
                p_θ(h_i | v) = Bernoulli(σ(ξ_i(v)))  # Eq. (13)

            Implementation:
                p_h = σ(ξ(v))  # element-wise
                h ~ Bernoulli(p_h)  # element-wise

        Note:
            - ξ(v) is defined in ``xi(...)``.
            - Compute p(h|v) = σ(Acos(x) + Bsin(x)).

        **Shapes**:
            Mathematical (paper):
                v: ``[nv]``
                ξ(v): ``[nh]``
                p_θ(h_i = 1 | v): Scalar

            Implementation:
                Input (v): ``[batch_size, nv]``
                ξ(v): ``[batch_size, nh]``
                p_h: ``[batch_size, nh]``
                Output (h): ``[batch_size, nh]``

        Args:
            v: Batch of visible layer state vectors.

        Returns:
            h: Batch of hidden layer state vectors.
        """
        p_h = torch.sigmoid(self.xi(v))
        return self.bernoulli_sampling(p_h)

    def h_to_v(self, h: torch.Tensor) -> torch.Tensor:
        """Sample a batch of visible states from a batch of hidden states.

        Computes inverse dispersion and mean angle from the hidden units
        and then samples visible units from the computed VonMises distribution.

        **Equations**:
            Mathematical (Paper, Section 3.6):
                κ_j = sqrt(α_j^2 + β_j^2) ≥ 0
                sin(μ_j) = β_j /κ_j
                cos(μ_j) = α_j /κ_j

            Implementation:
                kappa = sqrt(α(h)**2 + β(h)**2).clamp(min=1e-6, max=1e2)
                mu = atan2(β(h), α(h))
                v = D.VonMises(mu, kappa).sample()
                v = torch.remainder(v, 2 * torch.pi)

        Notes:
            - α(h) is defined in ``hW(...)``.
            - β(h) is defined in ``hW(...)``.
            - κ is the inverse dispersion and μ is the mean angle for the
                VonMises distribution.

        **Shapes**:
            Mathematical (paper):
                α(h): ``[nv]``
                β(h): ``[nv]``
                κ: ``[nv]``
                μ: ``[nv]``

            Implementation:
                Input (h): ``[batch_size, nh]``
                α(h): ``[batch_size, nv]``
                β(h): ``[batch_size, nv]``
                kappa: ``[batch_size, nv]``
                mu: ``[batch_size, nv]``
                Output (v): ``[batch_size, nv]``

        Args:
            h: Batch of hidden layer state vectors.

        Returns:
            v: Batch of visible layer state vectors.
        """
        alpha = self.hW(h, self.A)  # [batch_size, nv]
        beta = self.hW(h, self.B)  # [batch_size, nv]

        kappa = torch.sqrt(alpha**2 + beta**2).clamp(
            min=1e-6, max=1e2
        )  # [batch_size, nv]
        mu = torch.atan2(beta, alpha)

        v = D.VonMises(mu, kappa).sample()
        return torch.remainder(v, 2 * torch.pi)

    def langevin_update(
        self, v: torch.Tensor, epsilon: float = 0.1
    ) -> torch.Tensor:
        """This function is not implemented."""
        raise NotImplementedError(
            "Langevin dynamics for VonMises visibles not implemented."
        )

    def forward(
        self,
        v: torch.Tensor,
        mc: str = "gibbs",
        k: int = 1,
        epsilon: float = 0.1,
    ) -> torch.Tensor:
        """Performs k-step Gibbs sampling or k-step Langevin dynamics sampling.

        Performs k-step Gibbs sampling where one step computes v->h->v' or
        k-step Langebin dynamics sampling where one step computes v->v'.

        Args:
            v: Batch of visible layer state vectors.
            mc: String indictating the type of sampling, 'gibbs' or 'langevin'.
            k: Number of steps in k-step sampling.
            epsilon: Float used in Langevin dynamics denoting step-size.

        Returns:
            v: Batch of new visible layer state vectors after k-step sampling.
        """
        v = v.view(-1, self.n_visible)  # [batch_size, nv]

        if mc == "gibbs":
            with torch.no_grad():  # Gibbs does not need to do auto_diff
                for _ in range(k):
                    h = self.v_to_h(v)
                    v = self.h_to_v(h)

        elif mc == "langevin":  # Langevin MUST keep autograd to use it
            raise NotImplementedError

        return v  # .detach()

    def visible_energy(self, v: torch.Tensor) -> torch.Tensor:
        """Compute the visible energy E(v).

        The visible energy is computed from the visible units.

        **Equations**:
            Mathematical (Paper Section 3.6):
                E_θ(v) = -sum_{i=1}^{nh} Softplus(ξ_i(v))  # Eq. (62)

            Implementation:
                output = -sum(softplus(ξ(v)), dim=1))

        Note:
            ξ(v) is defined in ``xi(...)``.

        **Shapes**:
            Mathematical (Paper):
                v: ``[nv]``
                ξ(v): ``[nh]``
                E_θ(v): Scalar

            Implementation:
                Input (v): ``[batch_size, nv]``
                ξ(v): ``[batch_size, nh]``
                Output: ``[batch_size]``

        Args:
            v: Batch of visible layer state vectors.

        Returns:
            Batch of visible energies computed from v.
        """
        return -torch.sum(F.softplus(self.xi(v)), dim=1)  # [batch_size,]

    def contrastive_divergence(
        self,
        v0: torch.Tensor,
        pcd: bool = False,
        mc: str = "gibbs",
        k: int = 1,
        epsilon: float = 0.1,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        momentum: float = 0.0,
    ):
        """Perform gradient descent for one batch with k-step Contrastive Divergence.

        Performs gradient descent for a batch with either Contrastive
        Divergence (CD) or Persistent Contrastive Divergence (PCD)
        with Maximum likelihood estimation (MLE).

        Relevant Sections from the paper:
            - Section 2.3.1, Maximum likelihood estimation (MLE) and Kullback-Leibler (KL) divergence
            - Section 3.6, Periodic (angular) visible units with sin-cos activations: von Mises distribution
            - Section 2.4, Selected algorithms for stochastic sampling from the model distribution

        **Equations**:
            Mathematical (Paper):
                ∇_θL_MLE(θ) = ⟨∇_θE_θ(v)⟩_(v ~ pD) - ⟨∇_θE_θ(v)⟩_(v ~ pθ)  # Section 2.3.1, Eq. (17)
                dE_θ(v)/dA_ij = -cos(v_j)σ(ξ_i(v))  # Section 3.6, Eq. (64)
                dE_θ(v)/dB_ij = -sin(v_j)σ(ξ_i(v))  # Section 3.6, Eq. (64)
                dE_θ(v)/dc_i = -σ(ξ_i(v))  # Section 3.1, Eq. (38)

        Note:
            ξ(v) is defined in ``xi(...)``.

        **Shapes**:
            Mathematical (Paper):
                v: ``[nv]``
                ξ(v): ``[nh]``

        Args:
            v0: Visible data batch.
            pcd: Whether PCD is used.
            mc: String indictating the type of sampling, 'gibbs' or 'langevin'.
            k: Number of steps in k-step sampling.
            epsilon: Float used in Langevin dynamics denoting step-size.
            lr: Learning rate used for gradient updates.
            weight_decay: Weight decay rate for L2 regularization.
            momentum: Momentum coefficient used for trianing with momentum.

        Returns:
            tuple: A tuple containing:
                - E_data: Visible energy of data.
                - E_model: Visible energy of model.
                - E_diff: Difference between E_data and E_model.
                - MSE: Mean Square Error training metric.
                - ce: Cross Entropy Loss used as a training metric.
        """
        batch_size = v0.size(0)
        v_batch = v0.view(-1, self.n_visible)  # [batch_size, nv]

        # -------- Data term / positive phase --------
        with torch.no_grad():
            p_h_batch = torch.sigmoid(self.xi(v_batch))  # [batch_size, nh]

        self.A.grad = p_h_batch.T @ torch.cos(v_batch) / batch_size  # [nh, nv]
        self.B.grad = p_h_batch.T @ torch.sin(v_batch) / batch_size  # [nh, nv]
        self.h_bias.grad = torch.mean(p_h_batch, dim=0)  # [nh, ]

        # -------- Gibbs sampling / negative phase --------

        # Initialize persistent chain the first time
        if self.persistent_v is None:
            self.persistent_v = v_batch.detach().clone()

        if pcd == True:  # PCD
            self.persistent_v = self.persistent_v.detach()
            v_sample = self.forward(
                self.persistent_v, mc, k, epsilon
            )  # [batch_size, nv]
            self.persistent_v = v_sample.detach().clone()
        else:  # CD
            v_sample = self.forward(v_batch, mc, k, epsilon)  # [batch_size, nv]

        with torch.no_grad():
            p_h_sample = torch.sigmoid(self.xi(v_sample))  # [batch_size, nh]

        # data term - model term
        self.A.grad -= (
            p_h_sample.T @ torch.cos(v_sample) / batch_size
        )  # [nh, nv]
        self.B.grad -= (
            p_h_sample.T @ torch.sin(v_sample) / batch_size
        )  # [nh, nv]
        self.h_bias.grad -= p_h_sample.mean(dim=0)  # [nh, ]

        # Weight Decay: L2 Regularization
        self.A.grad -= weight_decay * self.A
        self.B.grad -= weight_decay * self.B

        # Calculate momentum, or delta W
        self.vA = momentum * self.vA + lr * self.A.grad.clone().detach()
        self.vB = momentum * self.vB + lr * self.B.grad.clone().detach()
        self.vh_bias = (
            momentum * self.vh_bias + lr * self.h_bias.grad.clone().detach()
        )

        # Update parameters manually by gradient descent
        with torch.no_grad():
            self.A += self.vA
            self.B += self.vB
            self.h_bias += self.vh_bias

        # -------- Diagnostics --------
        E_data = torch.mean(self.visible_energy(v_batch))
        E_model = torch.mean(self.visible_energy(v_sample))
        E_diff = E_model - E_data

        v_recon = self.forward(v_batch, mc="gibbs", k=1)
        MSE = torch.mean(
            (torch.cos(v_recon) - torch.cos(v_batch)) ** 2
            + (torch.sin(v_recon) - torch.sin(v_batch)) ** 2
        )  # Cos-Sin MSE

        return E_data, E_model, E_diff, MSE, torch.tensor([float("nan")])
