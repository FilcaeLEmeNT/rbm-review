import torch
import copy

def overwrite_train_cfg(train_cfg, schedule_node):
    train_cfg_overwritten = copy.deepcopy(train_cfg)

    for k in train_cfg_overwritten:
        if schedule_node.get(k) is not None:
            train_cfg_overwritten[k] = schedule_node.get(k)

    return train_cfg_overwritten
    
def train_cd(model, device, train_loader, train_cfg, n_epochs, starting_epoch = 0):
    """
    Train the RBM model using Contrastive Divergence or Persistent Contrastive Divergence.

    Parameters:
    - model: RBM model instance.
    - device: device outputted by get_device().
    - train_loader: DataLoader for training data.
    - train_cfg: contains parameters.
    - n_epochs: Number of epochs to train in this session.
    - starting_epoch: This is used to ensure that training is consistent with training.schedule even when training is resumed.

    Returns:
    - history: Dictionary with training metrics
    """
    # Load parameters based on how many epochs have been trained by the model.
    # Create a start_epoch: schedule_index dictionary for schedule.
    schedule = train_cfg.get("schedule", [{"start": 0}])
    epoch_to_idx = {schedule[i]["start"]: i for i in range(len(schedule))}

    # Iterate through schedule to find the right starting parameters.
    starting_idx = 0
    for k in list(epoch_to_idx.keys()):
        if starting_epoch >= k:
            starting_idx = epoch_to_idx.pop(k)

    current = overwrite_train_cfg(train_cfg, schedule[starting_idx])
    lr = current.get("lr")
    weight_decay = current.get("weight_decay")
    momentum = current.get("momentum")
    k = current.get("k")
    pcd = current.get("pcd")
    mc = current.get("mc")
    epsilon = current.get("epsilon")
    print(f"\nTraining with {"PCD" if pcd else "CD"} and {k}-step {mc} updates. lr={lr}, weight_decay={weight_decay}, momentum={momentum}, epsilon={epsilon}")
    
    history = {"E_data": [], "E_model": [], "E_diff": [], "mse": [], "loss": []}

    for epoch in range(n_epochs):
        if starting_epoch + epoch in epoch_to_idx:
            current = overwrite_train_cfg(train_cfg, schedule[epoch_to_idx[starting_epoch + epoch]])
            lr = current.get("lr")
            weight_decay = current.get("weight_decay")
            momentum = current.get("momentum")
            k = current.get("k")
            pcd = current.get("pcd")
            mc = current.get("mc")
            epsilon = current.get("epsilon")

            if not pcd and model.persistent_v is not None:
                model.persistent_v = None
                print("Deleted persistent batch")

            print(f"\nTraining with {"PCD" if pcd else "CD"} and {k}-step {mc} updates. lr={lr}, weight_decay={weight_decay}, momentum={momentum}, epsilon={epsilon}")

        E_data_epoch, E_model_epoch, E_diff_epoch, mse_epoch, loss_epoch = 0., 0., 0., 0., 0.
        for _, batch_data in enumerate(train_loader):
            X_train = batch_data[0] if isinstance(batch_data, list) else batch_data
            X_train = X_train.to(device)
            E_data, E_model, E_diff, mse,loss = model.contrastive_divergence(X_train, pcd, mc, k, epsilon, lr, weight_decay, momentum)
            E_data_epoch += E_data.item()
            E_model_epoch += E_model.item()
            E_diff_epoch += E_diff.item()
            mse_epoch += mse.item()
            loss_epoch += loss.item()

        E_data_epoch /= len(train_loader)
        E_model_epoch /= len(train_loader)
        E_diff_epoch /= len(train_loader)
        mse_epoch /= len(train_loader)
        loss_epoch /= len(train_loader)
        
        history["E_data"].append(E_data_epoch)
        history["E_model"].append(E_model_epoch)

        history["E_diff"].append(E_diff_epoch)
        history["mse"].append(mse_epoch)
        history["loss"].append(loss_epoch)

        print(f"Epoch {epoch + 1}/{n_epochs}, E_data: {E_data_epoch:.4f}, E_model: {E_model_epoch:.4f}, E_diff: {E_diff_epoch:.4f}, mse: {mse_epoch:.4f}, loss: {loss_epoch:.4f}")

    return history

def train_sm(model, device, train_loader, train_cfg, n_epochs, starting_epoch = 0):
    """
    Train the RBM model using Score-Matching.
    Parameters pcd, mc, k, and epsilon are only for
    diagnosis metrics calculation, not for training itself.

    Parameters:
    - model: RBM model instance
    - train_loader: DataLoader for training data
    - pcd: Boolean, True for PCD, False for CD
    - mc: MCMC method, 'gibbs' or 'langevin'
    - k: Number of MCMC steps
    - epsilon: Step size for Langevin dynamics
    - lr: Learning rate
    - n_epochs: Number of training epochs

    Returns:
    - history: Dictionary with training metrics
    """
    # Load parameters based on how many epochs have been trained by the model.
    # Create a start_epoch: schedule_index dictionary for schedule.
    schedule = train_cfg.get("schedule", [{"start": 0}])
    epoch_to_idx = {schedule[i]["start"]: i for i in range(len(schedule))}

    # Iterate through schedule to find the right starting parameters.
    starting_idx = 0
    for k in list(epoch_to_idx.keys()):
        if starting_epoch >= k:
            starting_idx = epoch_to_idx.pop(k)
            print(epoch_to_idx)

    current = overwrite_train_cfg(train_cfg, schedule[starting_idx])
    lr = current.get("lr")
    weight_decay = current.get("weight_decay")
    momentum = current.get("momentum")
    k = current.get("k")
    pcd = current.get("pcd")
    mc = current.get("mc")
    epsilon = current.get("epsilon")
    print(f"\nTraining with score matching. lr={lr}, weight_decay={weight_decay}, momentum={momentum}, epsilon={epsilon}")
    
    history = {"E_data": [], "E_model": [], "E_diff": [], "mse": [], "loss": []}

    optimizer = torch.optim.Adam([
    {"params": [model.W], "lr": lr, "weight_decay": weight_decay},
    {"params": [model.z], "lr": lr * 0.1, "weight_decay": 0.0},
    {"params": [model.v_bias, model.h_bias], "lr": lr, "weight_decay": 0.0},
])

    for epoch in range(n_epochs):
        if starting_epoch + epoch in epoch_to_idx:
            current = overwrite_train_cfg(train_cfg, schedule[epoch_to_idx[starting_epoch + epoch]])
            lr = current.get("lr")
            weight_decay = current.get("weight_decay")
            momentum = current.get("momentum")
            k = current.get("k")
            pcd = current.get("pcd")
            mc = current.get("mc")
            epsilon = current.get("epsilon")
            print(f"\nTraining with {"PCD" if pcd else "CD"} and {k}-step {mc} updates. lr={lr}, weight_decay={weight_decay}, momentum={momentum}, epsilon={epsilon}")
            
        E_data_epoch, E_model_epoch, E_diff_epoch, loss_epoch, mse_epoch = 0., 0., 0., 0., 0.
        for _, batch_data in enumerate(train_loader):
            X_train = batch_data[0] if isinstance(batch_data, list) else batch_data
            v = X_train.to(device)

            # Train
            optimizer.zero_grad()
            loss = model.score_matching_loss(v)
            loss.backward()
            optimizer.step()

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
            mse_epoch += mse.item()
            loss_epoch += loss.item()
            
        E_data_epoch /= len(train_loader)
        E_model_epoch /= len(train_loader)
        E_diff_epoch /= len(train_loader)
        mse_epoch /= len(train_loader)
        loss_epoch /= len(train_loader)
        
        history["E_data"].append(E_data_epoch)
        history["E_model"].append(E_model_epoch)
        history["E_diff"].append(E_diff_epoch)
        
        history["mse"].append(mse_epoch)
        history["loss"].append(loss_epoch)
        
        print(f"Epoch {epoch + 1}/{n_epochs}, E_data: {E_data_epoch:.4f}, E_model: {E_model_epoch:.4f}, E_diff: {E_diff_epoch:.4f}, mse: {mse_epoch:.4f}, loss: {loss_epoch:.4f}")
        print(f"Average z: ", model.z.mean())
    return history