import sys
import torch
import torch.nn.functional as F
import numpy as np
from DataSets import get_DataLoader, get_SampledDataLoader
from tqdm import tqdm
from utils import get_p_from_explicit_model, get_multi_encoder, calc_multinomial_BA, calc_noised_multinomial_BA, \
    get_alpha_matrix


def entropy_loss(probs):
    return -torch.sum(probs * torch.log(probs + 1e-10), dim=-1).mean()


def repulsion_loss(X, epsilon=1e-4):
    """Computes a repulsion loss that penalizes close probability vectors."""
    num_points = X.shape[0]
    loss = 0.0
    for i in range(num_points):
        for j in range(i + 1, num_points):  # Avoid duplicate pairs
            dist = torch.norm(X[i] - X[j], p=2)  # Euclidean distance
            loss += 1.0 / (dist**2 + epsilon)  # Penalize small distances
    return loss / (num_points * (num_points - 1))  # Normalize


def train_multinomial_BA(model, optimizer, input_dim, num_epochs=100, res=None, device="cpu",
                         use_regularization=False, entropy_weight=0.1, update_probs=1, alpha=0.5,
                         current_batch_size=2 ** 14, noise_alpha=None):

    print(f"{device = }")
    n_test = model.n_trials * res + 1
    if res is not None:
        if n_test is not None:
            print(f"note: {n_test = }")

    # Initialize metrics as a dictionary
    training_metrics = {
        "input_dim": input_dim,
        "num_epochs": num_epochs,
        "current_batch_size": current_batch_size,
        "n_trials": model.n_trials,
        "use_regularization": use_regularization,
        "entropy_weight": entropy_weight,
        "update_probs": update_probs,
        "alpha": alpha,
        "res": res,
        "n_test": n_test,
        "lr_list": [],
        "epoch_losses": [],
        "running_p": [],
        "running_model": [],
        "running_weights": [],
        "noise_alpha": noise_alpha
    }

    noise_matrix = None
    if noise_alpha is not None:
        noise_matrix = get_alpha_matrix(model.n_trials, noise_alpha)

    probabilities = [1 / input_dim] * input_dim  # Default: uniform distribution

    epoch_size = current_batch_size

    training_metrics["epoch_size"] = epoch_size
    data_loader = get_DataLoader(input_dim, current_batch_size, epoch_size, device, shuffle=True)

    # Wrap the epoch loop with tqdm
    pbar = tqdm(range(num_epochs), desc="Training Progress", leave=True, dynamic_ncols=True, file=sys.stdout)
    for epoch in pbar:
        model.train()  # Set the model to training mode

        running_loss = 0.0  # To accumulate the loss over the epoch

        for batch in data_loader:
            # Save the old parameters (before the update)
            old_params = [param.clone() for param in model.parameters()]

            optimizer.zero_grad()  # Clear the gradients

            recon_batch = model(batch)

            loss = F.cross_entropy(recon_batch, batch)

            if use_regularization:
                # Generate data (one-hot vectors for each class)
                data = torch.eye(input_dim)
                data = data.to(device)

                probs = model.encoder(data)
                entropy_reg = entropy_loss(probs)  # Entropy regularization
                repulsion_reg = repulsion_loss(probs)  # Euclidean regularization

                loss = loss - entropy_weight * entropy_reg + 5e-2 * entropy_weight * repulsion_reg  # Combined loss

            # Backward pass: Compute gradients
            loss.backward()

            # Update model parameters
            optimizer.step()

            # Accumulate loss
            running_loss += loss.item()

            # Compute the step size (parameter update norm)
            step_size = 0
            for old_param, new_param in zip(old_params, model.parameters()):
                step_size += (new_param - old_param).norm().item()
            training_metrics["lr_list"].append(step_size)

        if (epoch + 1) % update_probs == 0:

            mean_p = get_p_from_explicit_model(model, input_dim, device, multi=True)
            if noise_alpha is not None:
                C, new_probabilities = calc_noised_multinomial_BA(mean_p, model.n_trials, noise_matrix)
            else:
                C, new_probabilities = calc_multinomial_BA(mean_p, model.n_trials)
            probabilities = alpha * probabilities + (1 - alpha) * np.array(new_probabilities)
            data_loader = get_SampledDataLoader(input_dim, probabilities, current_batch_size, epoch_size, device)

            if (epoch + 1) / num_epochs > 1:
                entropy_weight = 0
            else:
                entropy_weight = entropy_weight * 0.95

        training_metrics["running_weights"].append(probabilities)

        # Average loss for the epoch
        epoch_loss = running_loss / len(data_loader)
        training_metrics["epoch_losses"].append(epoch_loss)
        training_metrics["running_p"].append(get_p_from_explicit_model(model, input_dim, device, multi=True))
        training_metrics["running_model"].append(get_multi_encoder(model, input_dim, device))

        # Update the progress bar and display the loss
        pbar.set_postfix_str(f'Loss={epoch_loss:.4f}')

    return training_metrics
