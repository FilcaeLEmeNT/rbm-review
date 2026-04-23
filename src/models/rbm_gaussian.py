import torch
import torch.nn as nn
import torch.nn.functional as F

class RBM_gaussian(nn.Module):
    """
    RBM with Gaussian visible units and binary hidden units
    variance parameter sigma_j is set to 1
    parameters: W, b, c
    learning with Markov Chain Monte Carlo methods
    CD-k or PCD-k 
    update parameters for one batch
    calculate data visible energy and model visible energy
    difference between data and model visible energies
    one-step reconstruction mse
    """
    def __init__(self, n_visible, n_hidden):
        super(RBM_Gaussian, self).__init__()
        self.n_visible = n_visible # nv
        self.n_hidden = n_hidden # nh

        # Model parameters
        self.W = nn.Parameter(torch.randn(n_hidden, n_visible) * 0.01) # (nh, nv)
        self.v_bias = nn.Parameter(torch.zeros(n_visible))  # (nv, )
        self.h_bias = nn.Parameter(torch.zeros(n_hidden))   # (nh, )

        #self.sigma2 = torch.ones(n_visible) # self.sigma2 = torch.exp(self.z)
        self.z = nn.Parameter(torch.zeros(n_visible)) # self.z = torch.log(self.sigma2)

        # Initialize persistent chain
        self.persistent_v = None

    def alpha(self, v):
        """
        v^T/sigma2^T W^T + c  
        input_shape: [batch_size, nv]
        output shape: [batch_size, nh]
        """
        return F.linear(v*torch.exp(-self.z), self.W, self.h_bias)
    
    def beta(self, h):
        """
        mean of Gaussian unit
        h^TW + b
        input_shape: [batch_size, nh]
        output shape: [batch_size, nv]
        """
        return F.linear(h, self.W.t(), self.v_bias)
        
    def bernoulli_sampling(self, p):
        """Sampling from a Bernoulli distribution with prob p."""
        return torch.bernoulli(p) #.clamp(0.0, 1.0)

    def v_to_h(self, v):
        """
        Compute p(h|v) = sigmoid(Wv + c)
        and sample v -> h
        input_shape: [batch_size, nv]
        output shape: [batch_size, nh]
        """
        p_h = torch.sigmoid(self.alpha(v))
        return self.bernoulli_sampling(p_h)

    def h_to_v(self, h):
        """
        Compute mean = h^TW + b
        and sample h -> v with Gaussian N(v|mean, sigma^2) 
        input_shape: [batch_size, nh]
        output shape: [batch_size, nv]
        """
        mean_v = self.beta(h)
        return mean_v + torch.randn_like(mean_v)*torch.exp(self.z/2.)

    def langevin_update(self, v, epsilon=0.1):
        """
        One Langevin step: v(t+1) = v(t) - epsilon^2/2 * dE/dv + epsilon*noise
        """
        v = v.view(-1, self.n_visible) # [batch_size, nv]
       
        # if to use auto_diff
        v = v.detach().clone().requires_grad_(True)
        E = self.visible_energy(v).sum()
        grad_v = torch.autograd.grad(E, v)[0] # dE/dv

        # if to calculate gradient dE/dv manually
        #grad_v = torch.exp(-self.z.t()) * (v-self.v_bias.t() - torch.sigmoid(self.alpha(v))@self.W) # [batch_size, nv]
        
        # Gaussian noise
        noise = torch.randn_like(v) # [batch_size, nv]

        # Langevin update
        v_new = v - epsilon**2/2. * grad_v + epsilon * noise

        # keep within (0,1)?
        #v_new = torch.clamp(v_new, 0.0, 1.0)
        return v_new #torch.sigmoid(v_new).detach()
        
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
            #v = self.bernoulli_sampling(v.detach())
                    
        return v #.detach()   

    def visible_energy(self, v):
        """Compute the visible energy E(v)."""
        vbias_term = torch.sum((v - self.v_bias)**2/2.0*torch.exp(-self.z), dim=1) # [batch_size, 1]
        wxv_c = self.alpha(v) # Wv + c: [batch_size, nh]
        hidden_term = torch.sum(F.softplus(wxv_c), dim=1) # [batch_size, 1]
        return vbias_term - hidden_term

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
        self.z.grad = -torch.mean(torch.exp(-self.z) * ((v_batch-self.v_bias)**2/2. - (p_h_batch@self.W)*v_batch), dim=0)
    
        # Gibbs sampling
        if pcd == True: # PCD
            self.persistent_v = self.persistent_v.detach()
            v_sample = self.forward(self.persistent_v, mc, k, epsilon)  # [batch_size, nv]
            self.persistent_v = v_sample.detach().clone()           
        else: # CD
            v_sample = self.forward(v_batch, mc, k, epsilon)  # [batch_size, nv]
        
        p_h_sample= torch.sigmoid(self.alpha(v_sample)) # [batch_size, nh]

        # data term - model term
        self.W.grad -= -torch.matmul(p_h_sample.t(), v_sample)/v_sample.size(0) # [nh, nv]
        self.h_bias.grad -= -torch.mean(p_h_sample, dim=0) # [nh]
        self.v_bias.grad -= -torch.mean(v_sample, dim=0) # [nv]
        self.z.grad -= -torch.mean(torch.exp(-self.z) * ((v_sample-self.v_bias)**2/2. - (p_h_sample@self.W)*v_sample), dim=0)

        # Update parameters manually by gradient descent
        for param in [self.W, self.v_bias, self.h_bias, self.z]:
            param.data -= lr * param.grad

        self.z.data.clamp_(-5, 5) # avoid langevin updates lead to exp(-z) explode
        
        E_data = torch.mean(self.visible_energy(v_batch))
        E_model = torch.mean(self.visible_energy(v_sample))       
        E_diff = E_model - E_data 
        #loss = nn.MSELoss(reduction='mean') 
        #MSE = loss(v_sample, v_batch) 
        
        v_recon = self.forward(v_batch, mc='gibbs', k=1)
        MSE = torch.mean((v_recon.clamp(0, 1) - v_batch)**2) # clamp v' into [0,1]
        
        return E_data, E_model, E_diff, MSE

    def gamma(self, v):
        """
        v_j - b_j - sum_{i=1}^{n_h}(sigmoid(alpha_i(v))W_ij)
        input_shape: [batch_size, nv]
        output shape: [batch_size, nv]
        """
        h_mean = torch.sigmoid(self.alpha(v)) # [batch_size, nh]
        summation = h_mean @ self.W 

        return v - self.v_bias.unsqueeze(0) - summation
        
    def sigmoid2(self, x):
        sigmoid = torch.sigmoid(x)
        return sigmoid * (1 - sigmoid)
    
    def score_matching_loss(self, v):
        precision = torch.exp(-self.z) # [nv, ]

        # compute first term, (precision * gamma)^2
        score = precision * self.gamma(v) # [batch_size, nv]
        score_norm_sq = (score ** 2).sum(dim=1) # [batch_size, ]

        # second term, precision/gaussian_lap
        gaussian_lap = precision.sum() # [1, ]

        # third term, hidden_lap
        h_var = self.sigmoid2(self.alpha(v)) # [batch_size, nh]
        
        # self.W [nh, nv] @ precision [nv, ]
        W_norm_sq = (self.W ** 2) @ (precision ** 2)   # [nh, ] Σ_j W_ij² e^{-2z_j}

        # h_var [batch_size, nh] * W_norm_sq [nh, ]
        hidden_lap = (h_var * W_norm_sq).sum(dim=1) # [batch_size, ]

        loss = 0.5 * score_norm_sq - gaussian_lap + hidden_lap
        return loss.mean()