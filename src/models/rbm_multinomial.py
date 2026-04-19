import torch
import torch.nn as nn
import torch.nn.functional as F

class RBM_multinomial(nn.Module):
    """
    RBM with categorical visible and binary hidden units
    parameters: W
    visible units v are represented by one-hot vectors
    learning with Markov Chain Monte Carlo methods
    CD-k or PCD-k 
    update parameters for one batch
    calculate data visible energy and model visible energy
    difference between data and model visible energies
    one-step reconstruction cross-entropy
    """
    def __init__(self, n_class, n_visible, n_hidden, mf=False):
        super(RBM_multinomial, self).__init__()
        self.n_visible = n_visible # nv
        self.n_hidden = n_hidden # nh
        self.n_class = n_class # C

        # Model parameters
        self.W = nn.Parameter(torch.randn(n_hidden, n_visible*n_class) * 0.01) # (nh, nv*C)
        self.v_bias = nn.Parameter(torch.zeros(n_visible*n_class))  # (nv*C, )
        self.h_bias = nn.Parameter(torch.zeros(n_hidden))   # (nh, )

        # Initialize persistent chain
        self.persistent_v = None
        self.mean_field = mf

    def xi(self, v):
        """
        v^T W^T + c
        input_shape: [batch_size, nv*C]
        output shape: [batch_size, nh]
        """
        return F.linear(v, self.W, self.h_bias)
    
    def omega(self, h):
        """
        h^TW + b
        input_shape: [batch_size, nh]
        output shape: [batch_size, nv*C]
        """
        return F.linear(h, self.W.t(), self.v_bias)
        
    def bernoulli_sampling(self, p):
        """Sampling from a Bernoulli distribution with prob p."""
        return torch.bernoulli(p)

    def v_to_h(self, v):
        """
        Compute p(h|v) = sigmoid(Wv + c)
        and sample v -> h
        input_shape: [batch_size, nv*C]
        output shape: [batch_size, nh]
        """
        p_h = torch.sigmoid(self.xi(v))
        return self.bernoulli_sampling(p_h)

    def h_to_v(self, h):
        """
        logits = h^TW + b # [batch_size, nv*C]
        probability p(v|h) = softmax(logits) -> # [batch_size*nv, C]
        multinomial distribution
        input_shape: [batch_size, nh]
        output shape: [batch_size, nv*C]
        """
        logits = self.omega(h) # [batch_size, nv*C]
        #logits = logits.view(-1, self.n_visible, self.n_class) # [batch_size, nv, C]
        logits = logits.view(-1, self.n_class) # [batch_size*nv, C]
        p_v = F.softmax(logits, dim=-1) # [batch_size*nv, C]
        #p_v_flat = p_v.view(-1, self.n_class) # [batch_size*nv, C]
        samples = torch.multinomial(p_v, num_samples=1) # categorical {0,...,C-1}, [batch_size*nv, 1]
        v_sample = F.one_hot(samples.squeeze(1), num_classes=self.n_class) # [batch_size*nv, C]
        v_sample = v_sample.view(-1, self.n_visible * self.n_class).float() # [batch_size, nv*C]
        #p_v = p_v.view(-1, self.n_visible * self.n_class) # [batch_size, nv*C]
        return v_sample

    #def h_to_v_binary(self, h):
        #if self.mean_field == False: # binary v=0,1
        #    return self.bernoulli_sampling(p_v)
        #else:  # mean-field 0<v<1      
        #    return p_v

    def langevin_update(self, v, epsilon=0.1):
        """
        One Langevin step: v(t+1) = v(t) - epsilon^2/2 * dE/dv + epsilon*noise
        """
        v = v.view(-1, self.n_visible*self.n_class) # [batch_size, nv]
       
        # if to use auto_diff
        #v = v.detach().clone().requires_grad_(True)
        #E = self.visible_energy(v).sum()
        #grad_v = torch.autograd.grad(E, v)[0] # dE/dv

        # if to calculate gradient dE/dv manually
        grad_v = -self.v_bias.t() - torch.sigmoid(self.xi(v)) @ self.W # [batch_size, nv]
        
        # Gaussian noise
        noise = torch.randn_like(v) # [batch_size, nv]

        # Langevin update
        v_new = v - (epsilon**2/2.) * grad_v + epsilon * noise

        # absorbing (0,1)
        return v_new.clamp(0,1) #torch.sigmoid(v_new).detach()
        
    def forward(self, v, mc='gibbs', k=1, epsilon=0.1):
        """
        Performs k-step Gibbs sampling v->h->v' or
        k-step Langevin dynamics sampling v->v'
        """
        v = v.view(-1, self.n_visible*self.n_class) # [batch_size, nv]
      
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
        vbias_term = v.mv(self.v_bias) # b^T v: [batch_size, nv*C] x [nv*C, 1] = [batch_size, 1]
        wxv_c = self.xi(v) # Wv + c: [batch_size, nh]
        hidden_term = torch.sum(F.softplus(wxv_c), dim=1) # [batch_size, 1]
        return -vbias_term - hidden_term

    def contrastive_divergence(self, v0, pcd=False, mc='gibbs', k=1, epsilon=0.1, lr=0.001):
        """
        Perform gradient descent for one batch
        with k-step Contrastive Divergence 
        """
        v_batch = v0.view(-1, self.n_visible*self.n_class) # [batch_size, nv*C] 
        p_h_batch = torch.sigmoid(self.xi(v_batch)) # [batch_size, nh]

        # Initialize persistent chain the first time
        if self.persistent_v is None:
            self.persistent_v = v_batch.detach().clone()

        # data term
        self.W.grad = -torch.matmul(p_h_batch.t(), v_batch)/v_batch.size(0) # [nh, nv]
        self.h_bias.grad = -torch.mean(p_h_batch, dim=0) # [nh]
        self.v_bias.grad = -torch.mean(v_batch, dim=0) # [nv]

        # Gibbs sampling
        if pcd == True: # PCD
            self.persistent_v = self.persistent_v.detach()
            v_sample = self.forward(self.persistent_v, mc, k, epsilon)  # [batch_size, nv]
            self.persistent_v = v_sample.detach().clone()           
        else: # CD
            v_sample = self.forward(v_batch, mc, k, epsilon)  # [batch_size, nv]
        
        p_h_sample= torch.sigmoid(self.xi(v_sample)) # [batch_size, nh]

        # data term - model term
        self.W.grad -= -torch.matmul(p_h_sample.t(), v_sample)/v_sample.size(0) # [nh, nv]
        self.h_bias.grad -= -torch.mean(p_h_sample, dim=0) # [nh]
        self.v_bias.grad -= -torch.mean(v_sample, dim=0) # [nv]

        # Update parameters manually by gradient descent
        for param in [self.W, self.v_bias, self.h_bias]:
            param.data -= lr * param.grad
        
        #self.W.data.clamp_(-3, 5)
        
        E_data = torch.mean(self.visible_energy(v_batch))
        E_model = torch.mean(self.visible_energy(v_sample))       
        E_diff = E_model - E_data 
        #loss = nn.MSELoss(reduction='mean') 
        #MSE = loss(v_sample, v_batch) 
        
        #v_recon = self.forward(v_batch, mc='gibbs', k=1)
        #MSE = torch.mean((v_recon - v_batch)**2)

        logits = self.omega(self.v_to_h(v_batch)) # [batch_size, nv*C]
        CE = F.cross_entropy(logits.view(-1, self.n_class), v_batch.view(-1, self.n_class)) # [batch_size*nv, C] -> 1
        
        return E_data, E_model, E_diff, CE