import torch

def train_cd(model, device, train_loader, pcd, mc, k, epsilon, lr, n_epochs):
    if pcd == True:
        print(f"Training with PCD and {k}-step {mc} updates")
    else:
        print(f"Training with CD and {k}-step {mc} updates")
    
    history = {"E_data": [], "E_model": [], "E_diff": [], "mse": []}

    for epoch in range(n_epochs):
        E_data_epoch, E_model_epoch, E_diff_epoch, mse_epoch = 0., 0., 0., 0.
        for batch, batch_data in enumerate(train_loader):
            X_train = batch_data[0] if type(batch_data) == list else batch_data
            X_train = X_train.to(device)
            E_data, E_model, E_diff, mse = model.contrastive_divergence(X_train, pcd, mc, k, epsilon, lr)
            E_data_epoch += E_data.item()
            E_model_epoch += E_model.item()
            E_diff_epoch += E_diff.item()
            mse_epoch += mse.item()
        
        E_data_epoch /= len(train_loader)
        E_model_epoch /= len(train_loader)
        E_diff_epoch /= len(train_loader)
        mse_epoch /= len(train_loader)
        
        history["E_data"].append(E_data_epoch)
        history["E_model"].append(E_model_epoch)

        history["E_diff"].append(E_diff_epoch)
        history["mse"].append(mse_epoch)
            
        print(f"Epoch {epoch + 1}/{n_epochs}, E_data: {E_data_epoch:.4f}, E_model: {E_model_epoch:.4f}, E_diff: {E_diff_epoch:.4f}, mse: {mse_epoch:.4f}")

    return history

def train_sm(model, device, train_loader, pcd, mc, k, epsilon, lr, n_epochs):
    print("Training with score matching")
    
    history = {"E_data": [], "E_model": [], "E_diff": [], "loss": [], "mse": []}

    optimizer = torch.optim.Adam([
    {"params": [model.W], "weight_decay": 1e-4},
    {"params": [model.z], "weight_decay": 1e-5},
    {"params": [model.v_bias, model.h_bias], "weight_decay": 0.0}
], lr=lr)

    for epoch in range(n_epochs):
        E_data_epoch, E_model_epoch, E_diff_epoch, loss_epoch, mse_epoch = 0., 0., 0., 0., 0.
        for batch, (X_train, _) in enumerate(train_loader):
            v = X_train.to(device)

            # Train
            optimizer.zero_grad()
            loss = model.score_matching_loss(v)
            loss.backward()
            optimizer.step()

            model.z.data.clamp_(-5, 5)

            # Compute Energy and MSE for diagnosis
            with torch.no_grad():
                # Initialize persistent chain the first time
                if model.persistent_v is None:
                    model.persistent_v = v.detach().clone()

                # Gibbs sampling
                if pcd == True: # PCD
                    model.persistent_v = model.persistent_v.detach()
                    v_sample = model.forward(model.persistent_v, mc, k, epsilon)  # [batch_size, nv]
                    model.persistent_v = v_sample.detach().clone()           
                else: # CD
                    v_sample = model.forward(v, mc, k, epsilon)  # [batch_size, nv]

                E_data = torch.mean(model.visible_energy(v))
                E_model = torch.mean(model.visible_energy(v_sample))
                E_diff = E_model - E_data 
                
                v_recon = model.forward(v, mc='gibbs', k=1)
                mse = torch.mean((v_recon.clamp(0, 1) - v)**2) # clamp v' into [0,1]

            E_data_epoch += E_data.item()
            E_model_epoch += E_model.item()
            E_diff_epoch += E_diff.item()
            loss_epoch += loss.item()
            mse_epoch += mse.item()
            
        E_data_epoch /= len(train_loader)
        E_model_epoch /= len(train_loader)
        E_diff_epoch /= len(train_loader)
        loss_epoch /= len(train_loader)
        mse_epoch /= len(train_loader)
        
        history["E_data"].append(E_data_epoch)
        history["E_model"].append(E_model_epoch)
        history["E_diff"].append(E_diff_epoch)
        
        history["loss"].append(loss_epoch)
        history["mse"].append(mse_epoch)
        
        print(f"Epoch {epoch + 1}/{n_epochs}, E_data: {E_data_epoch:.4f}, E_model: {E_model_epoch:.4f}, E_diff: {E_diff_epoch:.4f}, loss: {loss_epoch:.4f}, mse: {mse_epoch:.4f}")

    return history