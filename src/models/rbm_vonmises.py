import torch
import torch.nn as nn
import torch.nn.functional as F

class RBM_vonmises(nn.Module):
    """
    RBM with Von Mises visible units and binary hidden units
    variance parameter sigma_j is set to 1
    parameters: A, B
    learning with Markov Chain Monte Carlo methods
    CD-k or PCD-k 
    update parameters for one batch
    calculate data visible energy and model visible energy
    difference between data and model visible energies
    one-step reconstruction mse
    """
    def __init__(self, n_visible, n_hidden):
        super(RBM_vonmises, self).__init__()
        self.n_visible = n_visible # nv
        self.n_hidden = n_hidden # nh

        # Model parameters
        limit = 4.0 * math.sqrt(6.0 / (n_hidden + n_visible))
        
        self.A = nn.Parameter(torch.empty(n_hidden, n_visible).uniform_(-limit, limit)) # (nh, nv)
        self.B = nn.Parameter(torch.empty(n_hidden, n_visible).uniform_(-limit, limit)) # (nh, nv)
        self.h_bias = nn.Parameter(torch.zeros(n_hidden))   # (nh, )

        # Initialize persistent chain
        self.persistent_v = None

    def xi(self, x):
        """
        xi(x)= Acos(x) + Bsin(x)
        input_shape: [batch_size, nv]
        output_shape: [batch_size, nh]
        """
        cosx = torch.cos(x) # size [batch_size, nv]
        sinx = torch.sin(x) # size [batch_size, nv]
        
        return cosx @ self.A.T + sinx @ self.B.T + self.h_bias # size [batch_size, nh]

    def hW(self, h, W):
        """
        return alpha(h) or beta(h)
        alpha(h) = hA
        beta(h) = hB
        input_shape: [batch_size, nh]
        output_shape: [batch_size, nv]
        """
        # size of h is [batch_size, nh]
        # size of W is [nh, nv]
        hxW = h @ W
        return hxW
        
    def bernoulli_sampling(self, p):
        """Sampling from a Bernoulli distribution with prob p."""
        return torch.bernoulli(p)

    def v_to_h(self, v):
        """
        Compute p(h|v) = sigmoid(Acos(x) + Bsin(x))
        and sample v -> h
        input_shape: [batch_size, nv]
        output shape: [batch_size, nh]
        """
        p_h = torch.sigmoid(self.xi(v))
        return self.bernoulli_sampling(p_h)

    def h_to_v(self, h):
        """
        sample h -> v with VonMises Distribution
        input_shape: [batch_size, nh]
        output shape: [batch_size, nv]
        """
        alpha = self.hW(h, self.A) # [batch_size, nv]
        beta = self.hW(h, self.B) # [batch_size, nv]
        
        kappa = torch.sqrt(alpha**2 + beta**2).clamp(min=1e-6, max=1e2) # [batch_size, nv]
        mu = torch.atan2(beta, alpha)
        
        v = D.VonMises(mu, kappa).sample()
        return torch.remainder(v, 2 * torch.pi)

    def langevin_update(self, v, epsilon=0.1):
        """
        None
        """
        raise NotImplementedError("Langevin dynamics for Von Mises visibles not implemented.")
        
    def forward(self, v, mc='gibbs', k=1, epsilon=0.1):
        """
        Performs k-step Gibbs sampling v->h->v' or
        k-step Langevin dynamics sampling v->v'
        """
        v = v.view(-1, self.n_visible) # [batch_size, nv]
      
        if mc == 'gibbs':
            with torch.no_grad(): # Gibbs does not need to do auto_diff
                for _ in range(k):
                    h = self.v_to_h(v)
                    v = self.h_to_v(h)
                    
        elif mc == 'langevin': # Langevin MUST keep autograd to use it
            raise NotImplementedError
                    
        return v #.detach()   

    def visible_energy(self, v):
        """
        Compute the visible energy E(v).
        -sum_{i}^{n_h}ln(1 + e^{xi(x)}
        F.softplus(x) = ln(1 + e^x)
        -sum_{i}^{n_h} torch.softplus(xi(x))
        """
        wxv = self.xi(v) # [batch_size, nh]
        return -torch.sum(F.softplus(wxv), dim=1) # [batch_size,]

    def contrastive_divergence(self, v0, pcd=False, mc='gibbs', k=1, epsilon=0.1, lr=1e-3, weight_decay=1e-4):
        """
        Perform gradient descent for one batch
        with k-step Contrastive Divergence 
        """
        batch_size = v0.size(0)
        v_batch = v0.view(-1, self.n_visible) # [batch_size, nv]

        # -------- Data term / positive phase --------
        with torch.no_grad():
            p_h_batch = torch.sigmoid(self.xi(v_batch)) # [batch_size, nh]
            
        self.A.grad = p_h_batch.T @ torch.cos(v_batch) / batch_size # [nh, nv]
        self.B.grad = p_h_batch.T @ torch.sin(v_batch) / batch_size # [nh, nv]
        self.h_bias.grad = torch.mean(p_h_batch, dim = 0) # [nh, ]
        
        # -------- Gibbs sampling / negative phase --------
        
        # Initialize persistent chain the first time
        if self.persistent_v is None:
            self.persistent_v = v_batch.detach().clone()
            
        if pcd == True: # PCD
            self.persistent_v = self.persistent_v.detach()
            v_sample = self.forward(self.persistent_v, mc, k, epsilon)  # [batch_size, nv]
            self.persistent_v = v_sample.detach().clone()           
        else: # CD
            v_sample = self.forward(v_batch, mc, k, epsilon)  # [batch_size, nv]

        with torch.no_grad():
            p_h_sample = torch.sigmoid(self.xi(v_sample)) # [batch_size, nh]

        # data term - model term
        self.A.grad -= p_h_sample.T @ torch.cos(v_sample) / batch_size # [nh, nv]
        self.B.grad -= p_h_sample.T @ torch.sin(v_sample) / batch_size # [nh, nv]
        self.h_bias.grad -= p_h_sample.mean(dim = 0) # [nh, ]

        # Weight Decay
        self.A.grad -= weight_decay * self.A
        self.B.grad -= weight_decay * self.B
        
        # -------- Manual Parameter updates --------
        with torch.no_grad():
            for param in [self.A, self.B, self.h_bias]:
                param.data += lr * param.grad

        # -------- Diagnostics --------
        E_data = torch.mean(self.visible_energy(v_batch))
        E_model = torch.mean(self.visible_energy(v_sample))       
        E_diff = E_model - E_data
        with torch.no_grad():
            v_recon = self.forward(v_batch, mc='gibbs', k=1)
        MSE = torch.mean(
            (torch.cos(v_recon) - torch.cos(v_batch))**2 + 
            (torch.sin(v_recon) - torch.sin(v_batch))**2
        ) # Cos-Sin MSE
        
        return E_data, E_model, E_diff, MSE