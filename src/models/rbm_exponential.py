# SPDX-License-Identifier: MIT
# src/models/rbm_exponential.py

"""Module that defines the RBM_exponential class."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RBM_exponential(nn.Module):
    """RBM with bounded exponential visible units and binary hidden units variance parameter sigma_j is set to 1.

    This class implements an RBM which models the visible units as a random
    variable sampled from a truncated exponential distribution. It learns data
    distributions using Contrastive Divergence (CD), Persistent Contrastive
    Divergence (PCD), or Parallel Tempering (PT) with Maximum likelihood estimation (MLE).
    It supports both Gibbs sampling and Langevin dynamics for MCMC.

    Notes:
        - Contains a function to update parameters for just one batch.
        - Calculates data visible energy and model visible energy.
        - Difference between data and model visible energies is recorded.
        - One-step reconstruction mse is used as a training metric.

    Attributes:
        n_visible: Number of visisble units.
        n_hidden: Number of hidden units.
        W: Weight matrix between visible and hidden units.
        v_bias: Bias vector for visible units.
        h_bias: Bias vector for visible units.
        vW: Velocity/Momentum vector for Weight matrix updates when using momentum.
        vv_bias: Velocity/Momentum vector for visible unit bias updates when using momentum
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

    def __init__(self, n_visible: int, n_hidden: int, mf: bool = False):
        """Initiate the RBM_exponential class.

        Args:
            n_visible: Number of visisble units.
            n_hidden: Number of hidden units.
            mf: Whether Mean-field is used when computing visible units
                from hidden units (Currently unused).
        """
        super(RBM_exponential, self).__init__()
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
        v: torch.Tensor,
        v_bias: torch.Tensor,
        W: torch.Tensor,
        h_bias: torch.Tensor,
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
            v_bias: Bias vector for visible units.
            W: Weight matrix between visible and hidden units.
            h_bias: Bias vector for hidden units.

        Returns:
            Batch of visible energies computed from v.
        """
        v = v.reshape(-1, W.shape[1])  # Reshape to [batch_size, nv]
        vbias_term = v @ v_bias
        hidden_term = torch.sum(F.softplus(F.linear(v, W, h_bias)), dim=1)
        return -vbias_term - hidden_term

    def xi(self, v: torch.Tensor) -> torch.Tensor:
        """Compute the hidden pre-activation vector from the visible units.

        Note the implementation is equivalent to the formulation in the paper, but
        uses batched row vectors instead of individual column vectors. Hence,
        the matrix multiplication appears as ``vWᵀ`` rather than ``Wv``.

        **Equations**:
            Mathematical (Paper, Section 3.4):
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

        Computes probabilities using the hidden units and then samples
        visible units from a truncated exponential distribution. Standard
        cdf inversion method is used to sample from the distribution.

        Notes:
            - given a uniform r.v.  0<u<1;
            - sample r.v. x subject to f(x) = k exp(kx)/(exp(k)-1)

        **Equations**:
            Mathematical (Paper, Section 3.4):
                p_θ(v_j |h) = β_j(h)exp(β_j(h)v_j) /
                    exp(β_j(h)v_max) - exp(β_j(h)v_min)
                    , v_min ≤ v_j ≤ v_max.  # Eq. (48)

                u = F(x) ∈ [0, 1]
                x = F^-1(u) = (1 / k) * log[(exp(kx_max) - exp(kx_min))u + exp(kx_min)]

            Implementation:
                k = β(h) * β
                clip_k = torch.clamp(k, -50.0, 50.0)
                u = torch.rand_like(k)
                small_beta_mask = clip_k.abs() < 1e-12
                arg = ((exp(clip_k) - 1.0) * u + 1.0).clamp(1e-12, 1e12)
                v_exp = (1 / clip_k) * log(arg)
                v = torch.where(small_beta_mask, u, v_exp).clamp(0, 1)

        Notes:
            - β(h) is defined in ``beta(...)``.
            - β is defined as 1 / T, not to be confused with the beta(...)
                function.
            - v_min is set as 0 and v_max treated as 1.

        **Shapes**:
            Mathematical (paper):
                h: ``[nh]``
                β(h): ``[nv]``
                v: ``[nv]``
                v_min: Scalar
                v_max: Scalar
                p_θ(v_j |h): Scalar

            Implementation:
                Input (h): ``[batch_size, nh]``
                k: ``[batch_size, nv]``
                Output (v): ``[batch_size, nv]``

        Args:
            h: Batch of hidden layer state vectors.
            beta_temp: Inverse temperature, 1 / T used when training with
                temperature other than 1.

        Returns:
            v: Batch of visible layer state vectors.
        """
        k = self.beta(h) * beta_temp
        clip_k = torch.clamp(k, -50.0, 50.0)

        u = torch.rand_like(k)
        small_beta_mask = clip_k.abs() < 1e-12

        exp_b = torch.exp(clip_k)
        arg = (exp_b - 1.0) * u + 1.0
        arg = torch.clamp(arg, 1e-12, 1e12)
        v_exp = torch.log(arg) / clip_k  # + 1e-10)

        return torch.where(small_beta_mask, u, v_exp).clamp(0, 1)

    def langevin_update(
        self, v: torch.Tensor, epsilon: float = 0.1, beta_temp: float = 1
    ) -> torch.Tensor:
        """Perform Langevin dynamics to update the visible units.

        Perform one step of Lagevin dynamics to get new batch of visible units
        from the current, v -> v'.

        Absorbing boundaries: v(t+1).clamp(0,1)
            if v < 0  -> v = 0
            if v > 1  -> v = 1
        or Boundary reflection:
            if v < 0  -> v = -v
            if v > 1  -> v = 2 - v

        **Equations**:
            Mathematical (Paper):
                z ~ N(0, I)  # Gaussian noise vector, Section 2.4.1
                v(n+1) = v(n) - epsilon^2/2 * ∇_vE_θ(v(n)) + epsilon * z  # Section 2.4.1, Eq. (33)
                ∇_vE_θ(v) = -b - Wᵀσ(ξ(v))  # Section 3.2

            Implementation:
                grad_v = -bᵀ - σ(ξ(v)) @ W
                noise = torch.rand_like(v)
                v_new = v - (epsilon**2 / 2.0) * β * grad_v + epsilon * noise.
                torch.clamp(v_new, 0, 1)

        Notes:
            - Reflecting boundaries is not implemented and
            absorption is used instead.
            - β is defined as 1 / T, not to be confused with beta(...)
            functions defined in other RBM architectures.

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
            beta_temp: Inverse temperature, 1 / T used when training with
                temperature other than 1.

        Returns:
            v_new: Batch of new visible layer state vectors.
        """
        v = self._reshape_visible(v)

        # if to use auto_diff
        # v = v.detach().clone().requires_grad_(True)
        # E = self.visible_energy(v).sum() # sum over batch
        # grad_v = torch.autograd.grad(E, v)[0] # dE/dv

        # if to calculate gradient dE/dv manually
        # grad_v = -self.beta(self.v_to_h(v)) # -log p(v|h) approximation
        grad_v = (
            -self.v_bias.t() - torch.sigmoid(self.xi(v)) @ self.W
        )  # [batch_size, nv]

        # Gaussian noise
        noise = torch.randn_like(v)  # [batch_size, nv]

        # Langevin update
        v_new = v - (epsilon**2 / 2.0) * beta_temp * grad_v + epsilon * noise

        # ---- Reflecting boundaries ----
        # mask_low = v_new < 0     # reflect below 0
        # v_new = torch.where(mask_low, -v_new, v_new)
        # mask_high = v_new > 1 # reflect above 1
        # v_new = torch.where(mask_high, 2 - v_new, v_new)

        return v_new.clamp(0, 1).detach()  # torch.sigmoid(v_new).detach()

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
            with torch.no_grad():
                for _ in range(k):
                    h = self.v_to_h(v, beta_temp=beta_temp)
                    v = self.h_to_v(h, beta_temp=beta_temp)
        elif mc == "langevin":
            for _ in range(k):
                v = self.langevin_update(v, epsilon, beta_temp=beta_temp)

        return v

    def visible_energy(self, v: torch.Tensor) -> torch.Tensor:
        """Compute the visible energy E(v).

        The visible energy is computed from the visible units.

        **Equations**:
            Mathematical (Paper Section 3.4):
                E_θ(v) = -bᵀv - sum_{i=1}^{nh} Softplus(ξ_i(v))  # Section 3.1, Eq. (37)

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

        v = self._reshape_visible(v)
        vbias_term = v.mv(self.v_bias)
        hidden_term = torch.sum(
            F.softplus(self.xi(v)), dim=1
        )  # [batch_size, 1]
        return -vbias_term - hidden_term

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
                or self.pt_betas.device != self.W.device
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
        negative_phase_method: str | bool = "CD",
        mc: str = "gibbs",
        k: int = 1,
        epsilon: float = 0.1,
        lr: float = 0.001,
        weight_decay: float = 1e-4,
        momentum: float = 0.0,
        pt_n_chains: int = 15,
        pt_max_T: float = 5,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict,
    ]:
        """Perform gradient descent for one batch with k-step CD, PCD, or PT.

        Performs gradient descent for a batch with either Contrastive
        Divergence (CD), Persistent Contrastive Divergence (PCD), or
        Parallel Tempering (PT) with Maximum likelihood estimation (MLE).

        Relevant Sections from the paper:
            - Section 2.3.1, Maximum likelihood estimation (MLE) and Kullback-Leibler (KL) divergence
            - Section 2.4, Selected algorithms for stochastic sampling from the model distribution
            - 3.4, Bounded continuous visible units: truncated exponential distribution
            - 2.4.3 Acceleration: parallel tempering

        **Equations**:
            Mathematical (Paper):
                ∇_θL_MLE(θ) = ⟨∇_θE_θ(v)⟩_(v ~ pD) - ⟨∇_θE_θ(v)⟩_(v ~ pθ)  # Section 2.3.1, Eq. (17)
                dE_θ(v)/dW_ij = -v_jσ(ξ_i(v))  # Section 3.1, Eq. (38)
                dE_θ(v)/dc_i = -σ(ξ_i(v))  # Section 3.1, Eq. (38)
                dE_θ(v)/db_j = -v_j  # Section 3.1, Eq. (38)

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
            mc: String indicating the type of sampling, 'gibbs' or 'langevin'.
            k: Number of steps in k-step sampling.
            epsilon: Float used in Langevin dynamics denoting step-size.
            lr: Learning rate used for gradient updates.
            weight_decay: Weight decay rate for L2 regularization.
            momentum: Momentum coefficient used for training with momentum.
            pt_n_chains: Number of chains used for Parallel Tempering (PT).
            pt_max_T: Maximum temperature used for Parallel Tempering (PT).

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
        p_h_batch = torch.sigmoid(self.xi(v_batch))

        self._initialize_persistent_state(
            v_batch, negative_phase_method, pt_n_chains, pt_max_T
        )

        # data term
        self.W.grad = -torch.matmul(p_h_batch.t(), v_batch) / v_batch.size(
            0
        )  # [nh, nv]
        self.v_bias.grad = -torch.mean(
            v_batch, dim=0
        )  # [nv] (v_j-b_j) but b_j will cancel
        self.h_bias.grad = -torch.mean(p_h_batch, dim=0)  # [nh]

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
        self.W.grad -= -torch.matmul(p_h_sample.t(), v_sample) / v_sample.size(
            0
        )  # [nh, nv]
        self.v_bias.grad -= -torch.mean(v_sample, dim=0)  # [nv]
        self.h_bias.grad -= -torch.mean(p_h_sample, dim=0)  # [nh]

        # Weight Decay: L2 Regularization
        self.W.grad -= weight_decay * self.W

        # Calculate momentum, or delta W
        with torch.no_grad():
            self.vW.mul_(momentum).add_(self.W.grad, alpha=lr)
            self.vv_bias.mul_(momentum).add_(self.v_bias.grad, alpha=lr)
            self.vh_bias.mul_(momentum).add_(self.h_bias.grad, alpha=lr)

        # Update parameters manually by gradient descent
        with torch.no_grad():
            self.W -= self.vW
            self.v_bias -= self.vv_bias
            self.h_bias -= self.vh_bias

        self.W.data.clamp_(-3, 5)

        with torch.inference_mode():
            E_data = torch.mean(self.visible_energy(v_batch))
            E_model = torch.mean(self.visible_energy(v_sample))
            E_diff = E_model - E_data

            # loss = nn.MSELoss(reduction='mean')
            # MSE = loss(v_sample, v_batch)

            v_recon = self.forward(v_batch, mc="gibbs", k=1)
            MSE = torch.mean((v_recon - v_batch) ** 2)  # clamp v' into [0,1]

            hidden_mean = p_h_sample.mean().detach().cpu().item()
            grad_norms = {
                "W": float(self.W.grad.norm().detach().cpu()),
                "v_bias": float(self.v_bias.grad.norm().detach().cpu()),
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

    def pt_compute_betas(self, pt_n_chains=15, pt_max_T=5):
        """Compute inverse temperatures for Parallel Tempering."""
        import numpy as np

        temps = torch.tensor(
            np.geomspace(1.0, pt_max_T, pt_n_chains),
            dtype=self.W.dtype,
            device=self.W.device,
        )
        return 1.0 / temps

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
        """Perform a Metropolis swap between two chains in Parallel Tempering (PT)."""
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
