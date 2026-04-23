import torch
import torch.nn as nn
import torch.nn.functional as F

class RBM_exponential(nn.Module):
    """
    RBM with bounded exponential visible units and binary hidden units
    variance parameter sigma_j is set to 1
    parameters: W, b, c
    learning with Markov Chain Monte Carlo methods
    CD-k or PCD-k 
    update parameters for one batch
    calculate data visible energy and model visible energy
    difference between data and model visible energies
    one-step reconstruction mse
    """
    def __init__(self, n_visible, n_hidden, mf=False):
        super(RBM_exponential, self).__init__()
        self.n_visible = n_visible # nv
        self.n_hidden = n_hidden # nh

        # Model parameters
        self.W = nn.Parameter(torch.randn(n_hidden, n_visible) * 0.01) # (nh, nv)
        self.v_bias = nn.Parameter(torch.zeros(n_visible))  # (nv, )
        self.h_bias = nn.Parameter(torch.zeros(n_hidden))   # (nh, )
      
        # Initialize persistent chain
        self.persistent_v = None
    def alpha(self, v):
        """
        v^T/sigma^T W^T + c  
        input_shape: [batch_size, nv]
        output shape: [batch_size, nh]
        """
        return F.linear(v, self.W, self.h_bias)
    
    def beta(self, h):
        """
        h^TW + b 
        input_shape: [batch_size, nh]
        output shape: [batch_size, nv]
        """
        return F.linear(h, self.W.t(), self.v_bias)
        
    def bernoulli_sampling(self, p):
        """Sampling from a Bernoulli distribution with prob p."""
        return torch.bernoulli(p)

    def v_to_h(self, v):
        """
        Compute p(h|v) = sigmoid(Wv + c)
        and sample v -> h
        input_shape: [batch_size, nv]
        output shape: [batch_size, nh]
        """
        p_h = torch.sigmoid(self.alpha(v))
        #p_h = torch.clamp(p_h, 0.0, 1.0)
        return self.bernoulli_sampling(p_h)

    def h_to_v(self, h):
        """
        Compute beta = h^TW + b
        and sample h -> v following exponential distribution
        given a uniform r.v.  0<u<1;
        sample r.v. x subject to f(x) = k exp(kx)/(exp(k)-1)
        input_shape: [batch_size, nh]
        output shape: [batch_size, nv]
        """
        hxw_b = self.beta(h)
        clip_hxw_b = torch.clamp(hxw_b, -50.0, 50.0) # 50 better than 10

        u = torch.rand_like(hxw_b)

        small_beta_mask = clip_hxw_b.abs() < 1e-12

        exp_b  = torch.exp(clip_hxw_b)
        arg = (exp_b - 1.) * u + 1.
        arg = torch.clamp(arg, 1e-12, 1e12)
        v_exp = torch.log(arg) / clip_hxw_b # + 1e-10) #clip_hxw_b

        v = torch.where(small_beta_mask, u, v_exp)
       
        return v.clamp(0,1) 

    def langevin_update(self, v, epsilon=0.1):
        """
        One Langevin step: v(t+1) = v(t) - epsilon^2/2 * dE/dv + epsilon*noise

        Absorbing boundaries: v(t+1).clamp(0,1) 
            if v < 0  -> v = 0
            if v > 1  -> v = 1
        or Boundary reflection:
            if v < 0  -> v = -v
            if v > 1  -> v = 2 - v
        """
        v = v.view(-1, self.n_visible) # [batch_size, nv]
       
        # if to use auto_diff
        #v = v.detach().clone().requires_grad_(True)
        #E = self.visible_energy(v).sum() # sum over batch
        #grad_v = torch.autograd.grad(E, v)[0] # dE/dv

        # if to calculate gradient dE/dv manually
        #grad_v = -self.beta(self.v_to_h(v)) # -log p(v|h) approximation
        grad_v = -self.v_bias.t() - torch.sigmoid(self.alpha(v)) @ self.W # [batch_size, nv]
        
        # Gaussian noise
        noise = torch.randn_like(v) # [batch_size, nv]

        # Langevin update
        v_new = v - (epsilon**2/2.) * grad_v + epsilon * noise

        # ---- Reflecting boundaries ----
        #mask_low = v_new < 0     # reflect below 0
        #v_new = torch.where(mask_low, -v_new, v_new)
        #mask_high = v_new > 1 # reflect above 1
        #v_new = torch.where(mask_high, 2 - v_new, v_new)

        return v_new.clamp(0,1) #torch.sigmoid(v_new).detach()
        
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
            for _ in range(k):
                v = self.langevin_update(v, epsilon)
                                
        return v #.detach()   

    def visible_energy(self, v):
        """Compute the visible energy E(v)."""
        vbias_term = v.mv(self.v_bias) # b^T v: [batch_size, nv] x [nv, 1] = [batch_size, 1]
        #vbias_term = torch.sum(v * self.v_bias, dim=1)
        wxv_c = self.alpha(v) # Wv + c: [batch_size, nh]
        hidden_term = torch.sum(F.softplus(wxv_c), dim=1) # [batch_size, 1]
        return -vbias_term - hidden_term


    def contrastive_divergence(self, v0, pcd=False, mc='gibbs', k=1, epsilon=0.1, lr=0.001):
        """
        Perform gradient descent for one batch
        with k-step Contrastive Divergence 
        """
        v_batch = v0.view(-1, self.n_visible) # [batch_size, nv] 
        p_h_batch = torch.sigmoid(self.alpha(v_batch)) # [batch_size, nh]

        # Initialize persistent chain the first time
        if self.persistent_v is None:
            self.persistent_v = v_batch.detach().clone()

        # data term
        self.W.grad = -torch.matmul(p_h_batch.t(), v_batch)/v_batch.size(0) # [nh, nv]
        self.h_bias.grad = -torch.mean(p_h_batch, dim=0) # [nh]
        self.v_bias.grad = -torch.mean(v_batch, dim=0) # [nv] (v_j-b_j) but b_j will cancel

        # Gibbs sampling
        if pcd == True: # PCD
            self.persistent_v = self.persistent_v.detach()
            v_sample = self.forward(self.persistent_v, mc, k, epsilon)  # [batch_size, nv]
            self.persistent_v = v_sample.detach().clone()           
        else: # CD
            v_sample = self.forward(v_batch, mc, k, epsilon)  # [batch_size, nv]
        
        p_h_sample= torch.sigmoid(self.alpha(v_sample)) # [batch_size, nh]

        #self.W.grad = None
        #self.v_bias.grad = None
        #self.h_bias.grad = None
        
        # data term - model term
        self.W.grad -= -torch.matmul(p_h_sample.t(), v_sample)/v_sample.size(0) # [nh, nv]
        self.h_bias.grad -= -torch.mean(p_h_sample, dim=0) # [nh]
        self.v_bias.grad -= -torch.mean(v_sample, dim=0) # [nv]

        # Update parameters manually by gradient descent
        for param in [self.W, self.v_bias, self.h_bias]:
            param.data -= lr * param.grad
            
        self.W.data.clamp_(-3, 5)
        
        E_data = torch.mean(self.visible_energy(v_batch))
        E_model = torch.mean(self.visible_energy(v_sample))       
        E_diff = E_model - E_data 
        #loss = nn.MSELoss(reduction='mean') 
        #MSE = loss(v_sample, v_batch) 
        
        v_recon = self.forward(v_batch, mc='gibbs', k=1)
        MSE = torch.mean((v_recon - v_batch)**2) # clamp v' into [0,1]
        
        return E_data, E_model, E_diff, MSE