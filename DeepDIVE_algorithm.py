import torch
import torch.optim as optim
from model_classes import Multinomial_VAE, Noised_Multinomial_VAE
from train import train_multinomial_BA
from utils import save_metrics

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Test parameters
test_number = 1
date = 607
model_file = f'results/test_model_{date}_{test_number}_'
dict_file = f'results/training_metrics_{date}_{test_number}_'
running_p_file = f"results/running_p_{date}_{test_number}.pkl"
running_w_file = f"results/running_w_{date}_{test_number}.pkl"

# Model hyperparameters
hidden_dim = 128
hidden_amount = 1
tau = 0.01

# Channel parameters
latent_dim = 3  # Latent space (channel) dimension
n_trials = 10
init_n = 10

# training parameter
num_epochs = 1500
init_epochs = 500

# Visualization parameters
res = 10
n_test = n_trials*res + 1


def run_curriculum(iter=0, n_trials=12, input_dim=5, update_probs=1, phase_two=True, phase_one=True, alpha=None):
    if alpha is not None:
        noise_matrix = [[1, 0, 0], [0, 1, 0], [alpha, alpha, 1 - 2 * alpha]]
        model = Noised_Multinomial_VAE(input_dim, hidden_dim, hidden_amount, latent_dim, n_trials=init_n,
                                       noise_matrix=noise_matrix, tau=tau, device=device).to(device)
    else:
        model = Multinomial_VAE(input_dim, hidden_dim, hidden_amount, latent_dim, n_trials=init_n, tau=tau,
                                device=device).to(device)

    optimizer = optim.Adam(model.parameters(), lr=5e-3)
    training_metrics = None

    if phase_one:
        training_metrics = train_multinomial_BA(model, optimizer, input_dim, init_epochs, res, device, update_probs=10,
                                                use_regularization=True, entropy_weight=0.01, noise_alpha=alpha)

    if phase_two:
        for param in model.decoder.parameters():
            param.requires_grad = True

        model.update_n(n_trials)
        optimizer = optim.Adam(model.parameters(), lr=5e-3)
        training_metrics = train_multinomial_BA(model, optimizer, input_dim, num_epochs, res, device,
                                                update_probs=update_probs, noise_alpha=alpha)

    torch.save(model.state_dict(), model_file + str(input_dim) + '_' + str(iter) + '.pth')
    save_metrics(training_metrics, dict_file + str(input_dim) + '_' + str(iter) + '.pkl')
    # Save to a file
    print("Model saved.")

    return training_metrics["running_p"], training_metrics["running_weights"]


def run_list():
    num_repetitions = 1

    # d value - Number of categories/classes
    min_d = 2
    max_d = 10
    test_vec = [d for d in range(min_d, max_d + 1)]

    running_p_list = []
    running_w_list = []
    for test_val in test_vec:
        print(f"{test_val = }")
        running_p_inner_list = []
        running_w_inner_list = []
        for i in range(num_repetitions):
            running_p, running_w = run_curriculum(iter=i, n_trials=n_trials, input_dim=test_val,
                                                  update_probs=50, phase_two=True)
            running_p_inner_list.append(running_p)
            running_w_inner_list.append(running_w)
        running_p_list.append(running_p_inner_list)
        running_w_list.append(running_w_inner_list)

    save_metrics(running_p_list, file_path=running_p_file)
    save_metrics(running_w_list, file_path=running_w_file)


run_list()
