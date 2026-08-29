# SPDX-License-Identifier: MIT
# src/models/rbm_vonmises.py

"""Module that defines the RBM_vonmises class."""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D


class RBM_vonmises(nn.Module):
    """RBM with VonMises visible units and binary hidden units.

    This class implements an RBM which models the visible units as a random
    variable sampled from a von-Mises distribution. It learns data
    distributions using Contrastive Divergence (CD), Persistent Contrastive
    Divergence (PCD), or Parallel Tempering (PT) with Maximum likelihood estimation (MLE).
    It supports both Gibbs sampling and Langevin dynamics for MCMC.

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
        B: Weight matrix between visible and hidden units.
        h_bias: Bias vector for hidden units.
        vA: Velocity/Momentum vector for Weight matrix updates when using momentum.
        vB: Velocity/Momentum vector for visible unit bias updates when using momentum
        vh_bias: Velocity/Momentum vector for hidden unit bias updates when using momentum
        persistent_v: Persistent visible-state used for persistent
            contrastive divergence (PCD).
        persistent_v_pt: Persistent visible-state used for parallel tempering (PT).
        pt_index_history: Dictionary recording the position history of each original PT chain as chains are swapped.
        pt_chain_perm: List mapping each current PT chain position to its original chain index.
        persistent_v_pt_energy: Visible energy of each persistent PT chain, tracked separately for use in Metropolis swaps.
        pt_betas: Tensor containing the inverse temperature for each PT chain, with shape [pt_n_chains].
        pt_swap_acceptance: Running acceptance rate of PT chain swaps.
        pt_swap_attempts: Number of PT chain-swap attempts accumulated so far.
        pt_betas: A tensor of shape [pt_n_chains * batch_size, 1]. Inverse temperatures used during parallel tempering (PT)

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
        self.register_buffer("persistent_v", None)
        self.register_buffer("persistent_v_pt", None)
        self.pt_index_history = None
        self.pt_chain_perm = None
        self.persistent_v_pt_energy = None
        self.pt_betas = None
        self.pt_swap_acceptance = None
        self.pt_swap_attempts = None

    @staticmethod
    def visible_energy(
        v: torch.Tensor, A: torch.Tensor, B: torch.Tensor, h_bias: torch.Tensor
    ) -> torch.Tensor:
        """Compute the visible energy E(v) from arguments and visible units.

        This static method is used to compute the visible energy from the given
        arguments and visible units. It is equivalent to the instance method
        ``visible_energy(...)`` but does not require an instance of the class
        to be called. It is used in the Parallel Tempering (PT) algorithm to
        compute the visible energy of the persistent chains at different
        temperatures.

        Args:
            v: Batch of visible layer state vectors.
            A: Weight matrix between visible and hidden units.
            B: Weight matrix between visible and hidden units.
            h_bias: Bias vector for hidden units.

        Returns:
            Batch of visible energies computed from v.
        """
        cosx = torch.cos(v)  # size [batch_size, nv]
        sinx = torch.sin(v)  # size [batch_size, nv]
        xi = cosx @ A.T + sinx @ B.T + h_bias  # size [batch_size, nh]
        return -torch.sum(F.softplus(xi), dim=1)  # [batch_size,]

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

    def _reshape_visible(self, v: torch.Tensor) -> torch.Tensor:
        """Return visible states as a 2D tensor shaped [batch_size, nv*C]."""
        if v.dim() == 1:
            return v.reshape(1, -1)
        return v.reshape(-1, self.n_visible)

    def bernoulli_sampling(self, p: torch.Tensor) -> torch.Tensor:
        """Sampling from a Bernoulli distribution with prob p."""
        return torch.bernoulli(p)

    def v_to_h(self, v: torch.Tensor, beta_temp: float = 1) -> torch.Tensor:
        """Sample a batch of hidden states from a batch of visible states.

        Computes probabilities using the visible units and then samples
        hidden units from a Bernoulli distribution.

        **Equations**:
            Mathematical (Paper, Section 2.2):
                p_θ(h_i | v) = Bernoulli(σ(ξ_i(v)))  # Eq. (13)

            Implementation:
                p_h = σ(β(ξ(v)))  # element-wise
                h ~ Bernoulli(p_h)  # element-wise

        Note:
            - ξ(v) is defined in ``xi(...)``.
            - Compute p(h|v) = σ(Acos(x) + Bsin(x)).
            - β is defined as 1 / T, not to be confused with the beta(...)
            function.

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
            beta_temp: Inverse temperature, 1 / T used when training with
                temperature other than 1.

        Returns:
            h: Batch of hidden layer state vectors.
        """
        p_h = torch.sigmoid(beta_temp * self.xi(v))
        return self.bernoulli_sampling(p_h)

    def h_to_v(self, h: torch.Tensor, beta_temp: float = 1) -> torch.Tensor:
        """Sample a batch of visible states from a batch of hidden states.

        Computes inverse dispersion and mean angle from the hidden units
        and then samples visible units from the computed VonMises distribution.

        **Equations**:
            Mathematical (Paper, Section 3.6):
                κ_j = sqrt(α_j^2 + β_j^2) ≥ 0
                sin(μ_j) = β_j /κ_j
                cos(μ_j) = α_j /κ_j

            Implementation:
                norm = sqrt(α(h)**2 + β(h)**2 + 1e-8)
                kappa = (β * norm).clamp(min=1e-6, max=1e2)
                mu = atan2(β(h), α(h))
                v = D.VonMises(mu, kappa).sample()
                v = torch.remainder(v, 2 * torch.pi)

        Notes:
            - α(h) is defined in ``hW(...)``.
            - β(h) is defined in ``hW(...)``.
            - κ is the inverse dispersion and μ is the mean angle for the
            VonMises distribution.
            - β is defined as 1 / T, not to be confused with the beta(...)
            function.

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
            beta_temp: Inverse temperature, 1 / T used when training with
                temperature other than 1.

        Returns:
            v: Batch of visible layer state vectors.
        """
        alpha = self.hW(h, self.A)  # [batch_size, nv]
        beta = self.hW(h, self.B)  # [batch_size, nv]

        norm = torch.sqrt(alpha**2 + beta**2 + 1e-8)

        kappa = (beta_temp * norm).clamp(min=1e-6, max=1e2)  # [batch_size, nv]
        mu = torch.atan2(beta, alpha)

        v = D.VonMises(mu, kappa).sample()
        return torch.remainder(v, 2 * torch.pi)

    def langevin_update(
        self, v: torch.Tensor, epsilon: float = 0.1, beta_temp: float = 1
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
        beta_temp: float = 1,
    ) -> torch.Tensor:
        """Performs k-step Gibbs sampling or k-step Langevin dynamics sampling.

        Performs k-step Gibbs sampling where one step computes v->h->v' or
        k-step Langebin dynamics sampling where one step computes v->v'.

        Args:
            v: Batch of visible layer state vectors.
            mc: String indictating the type of sampling, 'gibbs' or 'langevin'.
            k: Number of steps in k-step sampling.
            epsilon: Float used in Langevin dynamics denoting step-size.
            beta_temp: Inverse temperature, 1 / T used when training with
                temperature other than 1.

        Returns:
            v: Batch of new visible layer state vectors after k-step sampling.
        """
        v = self._reshape_visible(v)

        if mc == "gibbs":
            for _ in range(k):
                h = self.v_to_h(v, beta_temp=beta_temp)
                v = self.h_to_v(h, beta_temp=beta_temp)

        elif mc == "langevin":  # Langevin MUST keep autograd to use it
            for _ in range(k):
                v = self.langevin_update(v, epsilon, beta_temp=beta_temp)

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

    def _initialize_persistent_state(
        self,
        v_batch: torch.Tensor,
        negative_phase_method: str,
        pt_n_chains: int,
        pt_max_T: float,
    ) -> None:
        """Initialize persistent chains used by PCD and PT."""
        self.pt_swap_acceptance = None
        self.pt_swap_attempts = None

        if negative_phase_method == "PCD":
            if (
                self.persistent_v is None
                or self.persistent_v.shape != v_batch.shape
            ):
                self.persistent_v = v_batch.detach().clone()

        elif negative_phase_method == "PT":
            if (
                self.persistent_v_pt is None
                or self.persistent_v_pt.shape[1:] != v_batch.shape
                or self.persistent_v_pt.shape[0] != pt_n_chains
            ):
                self.persistent_v_pt = (
                    v_batch.detach().unsqueeze(0).repeat(pt_n_chains, 1, 1)
                )

            if (
                self.pt_chain_perm is None
                or len(self.pt_chain_perm) != pt_n_chains
            ):
                self._pt_index_history_init(pt_n_chains)

            if (
                self.pt_betas is None
                or self.pt_betas.numel() != pt_n_chains
                or self.pt_betas.device != self.A.device
            ):
                self.pt_betas = self.pt_compute_betas(
                    pt_n_chains=pt_n_chains, pt_max_T=pt_max_T
                )
            self.pt_swap_attempts = 0
            self.pt_swap_acceptance = 0.0

    def _sample_negative_phase(
        self,
        v_batch: torch.Tensor,
        negative_phase_method: str,
        mc: str,
        k: int,
        epsilon: float,
        pt_n_chains: int,
        pt_max_T: float,
    ) -> torch.Tensor:
        """Produce the negative-phase visible state for the chosen sampling method."""
        batch_size = v_batch.shape[0]

        with torch.no_grad():
            if negative_phase_method == "CD":
                return self.forward(v_batch, mc, k, epsilon)

            if negative_phase_method == "PCD":
                self.persistent_v = self.persistent_v.detach()
                v_sample = self.forward(self.persistent_v, mc, k, epsilon)
                self.persistent_v = v_sample.detach()
                return v_sample

            if negative_phase_method == "PT":
                if self.pt_betas is None:
                    self.pt_betas = self.pt_compute_betas(
                        pt_n_chains=pt_n_chains, pt_max_T=pt_max_T
                    )
                beta_temps = (
                    self.pt_betas[:, None].repeat(1, batch_size).reshape(-1, 1)
                )
                self.persistent_v_pt = self.persistent_v_pt.detach()
                for _ in range(2):
                    chain_states = self.persistent_v_pt.reshape(
                        -1, self.n_visible
                    )
                    evolved_states = self.forward(
                        chain_states,
                        mc,
                        k,
                        epsilon,
                        beta_temp=beta_temps,
                    ).detach()
                    self.persistent_v_pt.copy_(
                        evolved_states.reshape(self.persistent_v_pt.shape)
                    )
                    self.persistent_v_pt_energy = (
                        self.visible_energy(evolved_states)
                        .reshape(pt_n_chains, batch_size)
                        .mean(dim=1)
                    )

                    swap_accepts = 0
                    for i in range(0, pt_n_chains - 1, 2):
                        if self.pt_metropolis_swap(i, i + 1):
                            swap_accepts += 1
                    self._pt_index_history_append()
                    for i in range(1, pt_n_chains - 1, 2):
                        if self.pt_metropolis_swap(i, i + 1):
                            swap_accepts += 1
                    self._pt_index_history_append()

                    if self.pt_swap_attempts is not None:
                        self.pt_swap_attempts += 2 * (
                            (pt_n_chains // 2) if pt_n_chains > 1 else 1
                        )
                        self.pt_swap_acceptance = (
                            self.pt_swap_acceptance
                            * (self.pt_swap_attempts - swap_accepts)
                            + swap_accepts
                        ) / self.pt_swap_attempts

                return self.persistent_v_pt[0]

            raise ValueError(
                f"Invalid negative_phase_method={negative_phase_method} given."
            )

    def train_batch(
        self,
        v0: torch.Tensor,
        negative_phase_method: str = "CD",
        mc: str = "gibbs",
        k: int = 1,
        epsilon: float = 0.1,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        momentum: float = 0.0,
        pt_n_chains: int = 15,
        pt_max_T: float = 5,
    ):
        """Perform gradient descent for one batch with k-step CD, PCD, or PT.

        Performs gradient descent for a batch with either Contrastive
        Divergence (CD), Persistent Contrastive Divergence (PCD), or
        Parallel Tempering (PT) with Maximum likelihood estimation (MLE).

        Relevant Sections from the paper:
            - Section 2.3.1, Maximum likelihood estimation (MLE) and Kullback-Leibler (KL) divergence
            - Section 3.6, Periodic (angular) visible units with sin-cos activations: von Mises distribution
            - Section 2.4, Selected algorithms for stochastic sampling from the model distribution
            - 2.4.3 Acceleration: parallel tempering

        **Equations**:
            Mathematical (Paper):
                ∇_θL_MLE(θ) = ⟨∇_θE_θ(v)⟩_(v ~ pD) - ⟨∇_θE_θ(v)⟩_(v ~ pθ)  # Section 2.3.1, Eq. (17)
                dE_θ(v)/dA_ij = -cos(v_j)σ(ξ_i(v))  # Section 3.6, Eq. (64)
                dE_θ(v)/dB_ij = -sin(v_j)σ(ξ_i(v))  # Section 3.6, Eq. (64)
                dE_θ(v)/dc_i = -σ(ξ_i(v))  # Section 3.1, Eq. (38)

                Swap acceptance probability = min(1, exp((1/T_m - 1/T_m+1)
                    (E_θ(v_(m)) - E_θ(v_(m+1))))  # Section 2.4.3, Eq. (35)
        Note:
            ξ(v) is defined in ``xi(...)``.

        **Shapes**:
            Mathematical (Paper):
                v: ``[nv]``
                ξ(v): ``[nh]``

        Args:
            v0: Visible data batch.
            negative_phase_method: Sampling method for the negative phase, either "CD", "PCD", or "PT".
            mc: String indictating the type of sampling, 'gibbs' or 'langevin'.
            k: Number of steps in k-step sampling.
            epsilon: Float used in Langevin dynamics denoting step-size.
            lr: Learning rate used for gradient updates.
            weight_decay: Weight decay rate for L2 regularization.
            momentum: Momentum coefficient used for trianing with momentum.
            pt_n_chains: Number of chains used for Parallel Tempering (PT) sampling.
            pt_max_T: Maximum temperature used for Parallel Tempering (PT) sampling.

        Returns:
            tuple: A tuple containing:
                - E_data: Visible energy of data.
                - E_model: Visible energy of model.
                - E_diff: Difference between E_data and E_model.
                - MSE: Mean Square Error training metric.
                - ce: Cross Entropy Loss used as a training metric.
                - diagnostics: dictionary of other diagnostic measures.
        """
        v_batch = self._reshape_visible(v0)
        batch_size = v_batch.size(0)
        p_h_batch = torch.sigmoid(self.xi(v_batch))  # [batch_size, nh]

        self._initialize_persistent_state(
            v_batch, negative_phase_method, pt_n_chains, pt_max_T
        )

        self.A.grad = (
            -torch.matmul(p_h_batch.t(), torch.cos(v_batch)) / batch_size
        )  # [nh, nv]
        self.B.grad = (
            -torch.matmul(p_h_batch.t(), torch.sin(v_batch)) / batch_size
        )  # [nh, nv]
        self.h_bias.grad = -torch.mean(p_h_batch, dim=0)  # [nh, ]

        v_sample = self._sample_negative_phase(
            v_batch,
            negative_phase_method,
            mc,
            k,
            epsilon,
            pt_n_chains,
            pt_max_T,
        )

        p_h_sample = torch.sigmoid(self.xi(v_sample))  # [batch_size, nh]

        # data term - model term
        self.A.grad -= (
            -torch.matmul(p_h_sample.t(), torch.cos(v_sample)) / batch_size
        )  # [nh, nv]
        self.B.grad -= (
            -torch.matmul(p_h_sample.t(), torch.sin(v_sample)) / batch_size
        )  # [nh, nv]
        self.h_bias.grad -= -torch.mean(p_h_sample, dim=0)  # [nh, ]

        # Weight Decay: L2 Regularization
        self.A.grad -= weight_decay * self.A
        self.B.grad -= weight_decay * self.B

        # Calculate momentum, or delta W
        with torch.no_grad():
            self.vA.mul_(momentum).add_(self.A.grad, alpha=lr)
            self.vB.mul_(momentum).add_(self.B.grad, alpha=lr)
            self.vh_bias.mul_(momentum).add_(self.h_bias.grad, alpha=lr)

        # Update parameters manually by gradient descent
        with torch.no_grad():
            self.A -= self.vA
            self.B -= self.vB
            self.h_bias -= self.vh_bias

        with torch.inference_mode():
            E_data = torch.mean(self.visible_energy(v_batch))
            E_model = torch.mean(self.visible_energy(v_sample))
            E_diff = E_model - E_data

            v_recon = self.forward(v_batch, mc="gibbs", k=1)
            MSE = torch.mean(
                (torch.cos(v_recon) - torch.cos(v_batch)) ** 2
                + (torch.sin(v_recon) - torch.sin(v_batch)) ** 2
            )  # Cos-Sin MSE

            hidden_mean = p_h_sample.mean().detach().cpu().item()
            grad_norms = {
                "A": float(self.A.grad.norm().detach().cpu()),
                "B": float(self.B.grad.norm().detach().cpu()),
                "h_bias": float(self.h_bias.grad.norm().detach().cpu()),
            }
            diagnostics = {
                "hidden_mean": hidden_mean,
                "visible_occupancy": None,
                "grad_norms": grad_norms,
                "pt_swap_acceptance": float(self.pt_swap_acceptance)
                if self.pt_swap_acceptance is not None
                else None,
            }

        return (
            E_data,
            E_model,
            E_diff,
            MSE,
            torch.tensor([float("nan")]),
            diagnostics,
        )

    def pt_compute_betas(
        self,
        pt_n_chains=15,
        pt_max_T=5,
    ) -> list:
        """Compute the inverse temperatures (betas) for Parallel Tempering (PT).

        Args:
            pt_n_chains: Number of chains used for Parallel Tempering (PT) sampling.
            pt_max_T: Maximum temperature used for Parallel Tempering (PT) sampling.

        Returns:
            list: A list of inverse temperatures (betas) for each chain.
        """
        temps = torch.tensor(
            np.geomspace(1.0, pt_max_T, pt_n_chains),
            dtype=self.A.dtype,
            device=self.A.device,
        )
        betas = 1.0 / temps
        return betas

    def _pt_index_history_init(self, pt_n_chains: int):
        """Initialize the index history for Parallel Tempering (PT)."""
        self.pt_chain_perm = list(range(pt_n_chains))
        self.pt_index_history = {str(i): [] for i in range(pt_n_chains)}
        self._pt_index_history_append()

    def _pt_index_history_append(self):
        """Append the current chain permutation to the index history."""
        if self.pt_index_history is None or self.pt_chain_perm is None:
            return
        for position, original_id in enumerate(self.pt_chain_perm):
            self.pt_index_history[str(original_id)].append(position)

    def pt_metropolis_swap(self, idx1: int, idx2: int) -> bool:
        """Perform a Metropolis swap between two chains in Parallel Tempering (PT).

        Args:
            idx1: Index of the first chain.
            idx2: Index of the second chain.

        Returns:
            bool: True if the swap was accepted, False otherwise.
        """
        beta_term = self.pt_betas[idx1] - self.pt_betas[idx2]
        with torch.inference_mode():
            dE = (
                self.persistent_v_pt_energy[idx1].mean()
                - self.persistent_v_pt_energy[idx2].mean()
            )
        probability = torch.exp(beta_term * dE).clamp(max=1)

        if torch.rand(()) < probability:
            tmp = self.persistent_v_pt[idx1].clone()
            self.persistent_v_pt[idx1].copy_(self.persistent_v_pt[idx2])
            self.persistent_v_pt[idx2].copy_(tmp)

            tmp = self.persistent_v_pt_energy[idx1].clone()
            self.persistent_v_pt_energy[idx1].copy_(
                self.persistent_v_pt_energy[idx2]
            )
            self.persistent_v_pt_energy[idx2].copy_(tmp)

            if self.pt_chain_perm is not None:
                self.pt_chain_perm[idx1], self.pt_chain_perm[idx2] = (
                    self.pt_chain_perm[idx2],
                    self.pt_chain_perm[idx1],
                )

            return True

        return False
