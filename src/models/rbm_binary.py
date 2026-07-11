# SPDX-License-Identifier: MIT
# src/models/rbm_binary.py

"""Module that defines the RBM_binary class."""

import torch
from torch import nn
import torch.nn.functional as F


class RBM_binary(nn.Module):
    """Restricted Boltzmann Machine with binary visible and hidden units.

    This class implements an RBM which models the visible units as a random
    variable sampled from a Bernoulli distribution. It learns binary data
    distributions using Contrastive Divergence (CD) or Persistent Contrastive
    Divergence (PCD) with Maximum likelihood estimation (MLE). It supports
    both Gibbs sampling and Langevin dynamics for MCMC.

    Notes:
        - Contains a function to update parameters for just one batch.
        - Calculates data visible energy and model visible energy.
        - Difference between data and model visible energies is recorded.
        - One-step reconstruction mse is used as a training metric.

    Attributes:
        n_visible: Number of visible units.
        n_hidden: Number of hidden units.
        W: Weight matrix between visible and hidden units.
        v_bias: Bias vector for visible units. Denoted as b in the paper.
        h_bias: Bias vector for visible units. Denoted as c in the paper.
        vW: Velocity/Momentum vector for Weight matrix updates when using momentum.
        vv_bias: Velocity/Momentum vector for visible unit bias updates when using momentum
        vh_bias: Velocity/Momentum vector for hidden unit bias updates when using momentum
        persistent_v: Persistent visible-state used for persistent
            contrastive divergence (PCD).
        mean_field: Whether to use mean-field updates for the visible units.

    **Reference**:
        Kai Zhang and Sora Sakai,
        *Restricted Boltzmann Machines in Physics: Concepts, Theories, and Applications*.
        Throughout this module, this work is referred to as "the paper".
    """

    def __init__(self, n_visible: int, n_hidden: int, mf: bool = False):
        """Initiate the RBM_binary class.

        Args:
            n_visible: Number of visible units.
            n_hidden: Number of hidden units.
            mf: Whether Mean-field is used when computing visible units
                from hidden units.
        """
        super(RBM_binary, self).__init__()
        self.n_visible = n_visible  # nv
        self.n_hidden = n_hidden  # nh

        # Model parameters
        self.W = nn.Parameter(
            torch.randn(n_hidden, n_visible) * 0.01
        )  # (nh, nv)
        self.v_bias = nn.Parameter(torch.zeros(n_visible))  # (nv, )
        self.h_bias = nn.Parameter(torch.zeros(n_hidden))  # (nh, )

        # Define momentums
        self.register_buffer("vW", torch.zeros_like(self.W))
        self.register_buffer("vv_bias", torch.zeros_like(self.v_bias))
        self.register_buffer("vh_bias", torch.zeros_like(self.h_bias))

        # Initialize persistent chain
        self.persistent_v = None
        self.mean_field = mf

    def xi(self, v: torch.Tensor) -> torch.Tensor:
        """Compute the hidden pre-activation vector from the visible units.

        Note the implementation is equivalent to the formulation in the paper, but
        uses batched row vectors instead of individual column vectors. Hence,
        the matrix multiplication appears as ``vWᵀ`` rather than ``Wv``.

        **Equations**:
            Mathematical (Paper, Section 3.1):
                ξ(v) = Wv + c

            Implementation:
                ξ(v) = vWᵀ + c

        **Shapes**:
            Mathematical (Paper):
                v: ``[nv]``
                W: ``[nh, nv]``
                c: ``[nh]``
                ξ(v): ``[nh]``

            Implementation:
                Input (v): ``[batch_size, nv]``
                W: ``[nh, nv]``
                c: ``[nh]``
                Output (ξ(v)): ``[batch_size, nh]``

        Args:
            v: Batch of visible layer state vectors.

        Returns:
            ξ(v): Batch of hidden pre-activation vectors.
        """
        return F.linear(v, self.W, self.h_bias)

    def beta(self, h: torch.Tensor) -> torch.Tensor:
        """Compute the visible pre-activation vector from the hidden units.

        Note the implementation is equivalent to the formulation in the paper, but
        uses batched row vectors instead of individual column vectors. Hence,
        the matrix multiplication appears as ``hW`` rather than ``Wᵀh``.

        **Equations**:
            Mathematical (Paper, Section 3.1):
                β(h) = b + Wᵀh

            Implementation:
                β(h) = hW + b

        **Shapes**:
            Mathematical (Paper):
                h: ``[nh]``
                W: ``[nh, nv]``
                b: ``[nv]``
                β(h): ``[nv]``

            Implementation:
                Input (h): ``[batch_size, nh]``
                W: ``[nh, nv]``
                b: ``[nv]``
                Output (β(h)): ``[batch_size, nv]``

        Args:
            h: Batch of hidden layer state vectors.

        Returns:
            β(h): Batch of visible pre-activation vectors.
        """
        return F.linear(h, self.W.t(), self.v_bias)

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
            ξ(v) is defined in ``xi(...)``.

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

        Computes probabilities using the hidden units and then samples
        visible units from a Bernoulli distribution. If using mean-field,
        then asign the visible units as the probability without sampling
        from a Bernoulli distribution, yielding continuous visible
        units in [0, 1].

        **Equations**:
            Mathematical (Paper):
                p(v_j = 1 | h) = σ(β_j(h))  # Section 3.1, Eq. (39)

                Without mean-field:
                    v_j ~ Bernoulli(p(v_j = 1 | h))  # Section 3.1
                With mean-field:
                    v_j = σ(β_j(h))  # Section 3.2, Eq. (40)

            Implementation:
                p_v = σ(β(h))  # element-wise

                if mean_field:
                    v = p_v  # If mean_field = True
                else:
                    v ~ Bernoulli(p_v)  # element-wise

        Note:
            β(h) is defined in ``beta(...)``.

        **Shapes**:
            Mathematical (paper):
                h: ``[nh]``
                β(h): ``[nv]``
                p(v_j = 1 | h): Scalar
                v_j: Scalar

            Implementation:
                Input (h): ``[batch_size, nh]``
                β(h): ``[batch_size, nv]``
                p_v: ``[batch_size, nv]``
                Output (v): ``[batch_size, nv]``

        Args:
            h: Batch of hidden layer state vectors.

        Returns:
            v: Batch of visible layer state vectors.
        """
        p_v = torch.sigmoid(self.beta(h))

        if self.mean_field == False:  # binary v=0,1
            return self.bernoulli_sampling(p_v)
        else:  # mean-field 0<v<1
            return p_v

    def langevin_update(
        self, v: torch.Tensor, epsilon: float = 0.1
    ) -> torch.Tensor:
        """Perform Langevin dynamics to update the visible units.

        Perform one step of Lagevin dynamics to get new batch of visible units
        from the current, v -> v'.

        **Equations**:
            Mathematical (Paper):
                z ~ N(0, I)  # Gaussian noise vector, Section 2.4.1
                v(n+1) = v(n) - epsilon^2/2 * ∇_vE_θ(v(n)) + epsilon * z  # Section 2.4.1, Eq. (33)
                ∇_vE_θ(v) = -b - Wᵀσ(ξ(v))  # Section 3.2

            Implementation:
                grad_v = -bᵀ - σ(ξ(v)) @ W
                noise = torch.rand_like(v)
                v_new = v - (epsilon**2 / 2.0) * grad_v + epsilon * noise.
                torch.clamp(v_new, 0, 1)

        Note:
            ξ(v) is defined in ``xi(...)``.

        **Shapes**:
            Mathematical (Paper):
                v(n): ``[nv]``
                ξ(v): ``[nh]``
                W: ``[nh, nv]``
                b: ``[nv]``
                ∇_vE_θ(v): ``[nv]``
                v(n+1): ``[nv]``
                z: ``[nv]``

            Implementation:
                Input (v): ``[batch_size, nv]``
                ξ(v): ``[batch_size, nh]``
                W: ``[nh, nv]``
                b: ``[nv]``
                grad_v: ``[batch_size, nv]``
                Output (v_new): ``[batch_size, nv]``

        Args:
            v: Batch of visible layer state vectors.
            epsilon: Float used in Langevin dynamics denoting step-size.

        Returns:
            v_new: Batch of new visible layer state vectors.
        """
        v = v.view(-1, self.n_visible)  # [batch_size, nv]

        # if to use auto_diff
        # v = v.detach().clone().requires_grad_(True)
        # E = self.visible_energy(v).sum()
        # grad_v = torch.autograd.grad(E, v)[0]  # dE/dv

        # if to calculate gradient dE/dv manually
        grad_v = (
            -self.v_bias.t() - torch.sigmoid(self.xi(v)) @ self.W
        )  # [batch_size, nv]

        # Gaussian noise
        noise = torch.randn_like(v)  # [batch_size, nv]

        # Langevin update
        v_new = v - (epsilon**2 / 2.0) * grad_v + epsilon * noise

        # absorbing (0,1)
        return v_new.clamp(0, 1).detach()  # torch.sigmoid(v_new).detach()

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
            for _ in range(k):
                v = self.langevin_update(v, epsilon)
            # v = self.bernoulli_sampling(v.detach())

        return v  # .detach()

    def visible_energy(self, v: torch.Tensor) -> torch.Tensor:
        """Compute the visible energy E(v).

        The visible energy is computed from the visible units.

        **Equations**:
            Mathematical (Paper Section 3.1):
                E_θ(v) = -bᵀv - sum_{i=1}^{nh} Softplus(ξ_i(v))  # Eq. (37)

            Implementation:
                vbias_term = vb
                hidden_term = sum(softplus(ξ(v)), dim=1))
                output = -vbias_term - hidden_term

        Note:
            ξ(v) is defined in ``xi(...)``.

        **Shapes**:
            Mathematical (Paper):
                v: ``[nv]``
                b: ``[nv]``
                ξ(v): ``[nh]``
                E_θ(v): Scalar

            Implementation:
                Input (v): ``[batch_size, nv]``
                b: ``[nv]``
                vbias_term: ``[batch_size]``
                ξ(v): ``[batch_size, nh]``
                hidden_term: ``[batch_size]``
                Output: ``[batch_size]``

        Args:
            v: Batch of visible layer state vectors.

        Returns:
            Batch of visible energies computed from v.
        """
        vbias_term = v.mv(self.v_bias)
        hidden_term = torch.sum(F.softplus(self.xi(v)), dim=1)
        return -vbias_term - hidden_term

    def contrastive_divergence(
        self,
        v0: torch.Tensor,
        pcd: bool = False,
        mc: str = "gibbs",
        k: int = 1,
        epsilon: float = 0.1,
        lr: float = 0.001,
        weight_decay: float = 1e-4,
        momentum: float = 0.0,
    ) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        """Perform gradient descent for one batch with k-step Contrastive Divergence.

        Performs gradient descent for a batch with either Contrastive
        Divergence (CD) or Persistent Contrastive Divergence (PCD)
        with Maximum likelihood estimation (MLE).

        Relevant Sections from the paper:
            - Section 2.3.1, Maximum likelihood estimation (MLE) and Kullback-Leibler (KL) divergence
            - Section 3.1, Binary (or polar) visible units: Bernoulli (Rademacher) distribution
            - Section 2.4, Selected algorithms for stochastic sampling from the model distribution

        **Equations**:
            Mathematical (Paper):
                ∇_θL_MLE(θ) = ⟨∇_θE_θ(v)⟩_(v ~ pD) - ⟨∇_θE_θ(v)⟩_(v ~ pθ)  # Section 2.3.1, Eq. (17)
                dE_θ(v)/dW_ij = -v_jσ(ξ_i(v))  # Section 3.1, Eq. (38)
                dE_θ(v)/dc_i = -σ(ξ_i(v))  # Section 3.1, Eq. (38)
                dE_θ(v)/db_j = -v_j  # Section 3.1, Eq. (38)

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
        v_batch = v0.view(-1, self.n_visible)  # [batch_size, nv]
        p_h_batch = torch.sigmoid(self.xi(v_batch))  # [batch_size, nh]

        # Initialize persistent chain the first time
        if self.persistent_v is None:
            self.persistent_v = v_batch.detach().clone()

        # data term
        self.W.grad = -torch.matmul(p_h_batch.t(), v_batch) / v_batch.size(
            0
        )  # [nh, nv]
        self.v_bias.grad = -torch.mean(v_batch, dim=0)  # [nv]
        self.h_bias.grad = -torch.mean(p_h_batch, dim=0)  # [nh]

        # Gibbs sampling
        if pcd == True:  # PCD
            self.persistent_v = self.persistent_v.detach()
            v_sample = self.forward(
                self.persistent_v, mc, k, epsilon
            )  # [batch_size, nv]
            self.persistent_v = v_sample.detach().clone()
        else:  # CD
            v_sample = self.forward(v_batch, mc, k, epsilon)  # [batch_size, nv]

        p_h_sample = torch.sigmoid(self.xi(v_sample))  # [batch_size, nh]

        # data term - model term
        self.W.grad -= -torch.matmul(p_h_sample.t(), v_sample) / v_sample.size(
            0
        )  # [nh, nv]
        self.v_bias.grad -= -torch.mean(v_sample, dim=0)  # [nv]
        self.h_bias.grad -= -torch.mean(p_h_sample, dim=0)  # [nh]

        # Weight Decay: L2 Regularization
        self.W.grad -= weight_decay * self.W

        # Calculate momentum, or delta W
        self.vW = momentum * self.vW + lr * self.W.grad.clone().detach()
        self.vv_bias = (
            momentum * self.vv_bias + lr * self.v_bias.grad.clone().detach()
        )
        self.vh_bias = (
            momentum * self.vh_bias + lr * self.h_bias.grad.clone().detach()
        )

        # Update parameters manually by gradient descent
        with torch.no_grad():
            self.W -= self.vW
            self.v_bias -= self.vv_bias
            self.h_bias -= self.vh_bias

        # self.W.data.clamp_(-3, 5)

        E_data = torch.mean(self.visible_energy(v_batch))
        E_model = torch.mean(self.visible_energy(v_sample))
        E_diff = E_model - E_data

        with torch.no_grad():
            # MSE Loss
            # loss = nn.MSELoss(reduction='mean')
            # MSE = loss(v_sample, v_batch)

            v_recon = self.forward(v_batch, mc="gibbs", k=1)
            MSE = torch.mean((v_recon - v_batch) ** 2)

            # Cross Entropy Loss
            h_prob = torch.sigmoid(self.xi(v_batch))  # p(h | v_data)
            h_sample = torch.bernoulli(h_prob)  # sample h ~ p(h | v_data)
            v_prob = torch.sigmoid(self.beta(h_sample))  # p(v' | h)
            ce = nn.BCELoss()(v_prob, v_batch)

        return E_data, E_model, E_diff, MSE, ce
