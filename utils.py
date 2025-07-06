import pickle
import matplotlib.pyplot as plt
import itertools
import operator
import numpy as np
import torch
from scipy.special import factorial
from scipy.stats import multinomial
from sympy.utilities.iterables import multiset_permutations
from blahut_arimoto import blahut_arimoto
from sys import float_info


# ### Files ### #
def save_metrics(my_dict, file_path="training_metrics.pkl"):
    with open(file_path, "wb") as f:
        pickle.dump(my_dict, f)


def load_metrics(file_path="training_metrics.pkl"):
    # Load from a file
    with open(file_path, "rb") as f:
        loaded_results = pickle.load(f)
    return loaded_results


def restore_file(file_path):
    # Restore the Figure object from the file
    with open(file_path, 'rb') as f:
        fig = pickle.load(f)

    return fig


def plot_restored_file(file_path, block=True):
    f1 = restore_file(file_path)
    plt.show(block=block)


def plot_from_fig(f, sub=1, label=None, color=None):
    ax = f.get_axes()[sub]
    line = ax.get_lines()[0]
    x, y = line.get_data()
    if label is None:
        plt.plot(x, y)
    else:
        if color is None:
            plt.plot(x, y, label=label, color='teal')
        else:
            plt.plot(x, y, label=label, color=color)


# ### Information ### #
def calc_information(p_y_x, r):
    eps = float_info.epsilon
    log_base = 2
    # The number of inputs: size of |X|
    m = p_y_x.shape[0]
    # The number of outputs: size of |Y|
    n = p_y_x.shape[1]

    q = (np.array([r])).T * p_y_x
    q = q / (np.sum(q, axis=0)+eps)

    c = 0
    for i in range(m):
        if r[i] > 0:
            c += np.sum(r[i] * p_y_x[i, :] *
                        np.log(q[i, :] / r[i] + 1e-16))
    c = c / np.log(log_base)
    return c


def kl_divergence(p, q):
    eps = 1e-16
    return np.sum(np.where(p != 0, p * np.log2(eps + p / (q + eps)), 0))


def point_kl(n, x, p_y_k):
    p = calc_multinomial_channel(np.array([x]), n)
    return kl_divergence(p, p_y_k)


# r - number of balls
def combinations_with_replacement_counts(r, n):
    size = n + r - 1
    for indices in itertools.combinations(range(size), r-1):
        starts = [0] + [index+1 for index in indices]
        stops = indices + (size,)
        yield np.array(list(map(operator.sub, stops, starts)))


def multinomial_coeff(c):
    return factorial(c.sum()) / factorial(c).prod()


def multinomial_p(n, x_vec, y_vec):
    assert sum(y_vec) == n
    if np.abs(sum(x_vec)) - 1 > 1e-4 :
        x_vec = x_vec/sum(x_vec)
    assert len(x_vec) == len(y_vec)
    power_vec = x_vec ** y_vec
    ret_val = multinomial_coeff(y_vec)*(power_vec.prod())
    if ret_val < 0:
        ret_val = 0
    return ret_val


def calc_multinomial_channel(x, n):
    dim = len(x[0])
    p_y_x = []

    for symbol in x:
        p_y_symbol = [multinomial_p(n, symbol, permutation) for permutation in combinations_with_replacement_counts(dim, n)]
        p_y_symbol = p_y_symbol/sum(p_y_symbol)
        p_y_x.append(p_y_symbol)
    return p_y_x


def return_full(x):
    full = []
    for point in x:
        full.extend([np.array(p) for p in multiset_permutations(point)])
    return [np.array(a) for a in np.unique(full, axis=0)]


def calc_full_multinomial_channel(x, n):
    return calc_multinomial_channel(return_full(x), n)


def return_2_corners(dim):
    corners = []

    g = [1]
    g.extend([0] * (dim - 1))
    corners.append(np.array(g))

    g = [1 / dim] * dim
    corners.append(np.array(g))
    return np.unique(corners, axis=0)


def return_corners(dim):
    corners = []
    for d in range(1, dim + 1):
        g = [1 / d] * d
        g.extend([0] * (dim - d))
        corners.append(np.array(g))
    return np.unique(corners, axis=0)


def calc_pyk(x, n, extent=True):
    if extent:
        p = calc_full_multinomial_channel(x, n)
    else:
        p = calc_multinomial_channel(x, n)
    I, r = blahut_arimoto(np.asarray(p))
    p_y_k = np.matmul(np.array([r]), p)
    return p_y_k, I, r


def point_kl(n, x, p_y_k):
    p = calc_multinomial_channel(np.array([x]), n)
    return kl_divergence(p, p_y_k)


def calc_multinomial_BA_input_information(x, n):
    C, _ = calc_multinomial_BA(x, n)
    return C


def calc_multinomial_BA(x, n):
    p = calc_multinomial_channel(x, n)
    C, r = blahut_arimoto(np.asarray(p))
    return C, r


def calc_multinomial_information(x, r, n):
    assert len(x) == len(r)
    p_y_x = calc_multinomial_channel(x, n)
    C = calc_information(np.asarray(p_y_x), r)
    return C


def calc_equal_input_multinomial_information(x, n):
    m = len(x)
    r = np.array([1/m]*m)
    return calc_multinomial_information(x, r, n)


def fixed_multinomial_support_information(support, n_vec):
    return [calc_multinomial_BA_input_information(support, n) for n in n_vec]


# Load Information
def print_list_shape(running_p_list):
    amount_d = len(running_p_list)
    reps = len(running_p_list[0])
    epochs = len(running_p_list[0][0])
    dimension = len(running_p_list[0][0][0])
    print(f"{amount_d = }, {reps = }, {epochs = }, {dimension = }")


def load_support_information_multinomial(use_BA=False, file=None, take_last=1, use_BA_information=None, verbose=1):

    if file is not None:
        running_p_array = np.load(file)
    else:
        file1 = 'results/running_p_multi_BA_2d_5clean_0303.npy'
        file2 = 'results/running_p_multi_BA_2k_5d_clean_0803.npy'
        file3 = 'results/running_p_multi_BA_2k_5d_16_clean_0803.npy'

        running_p_array = np.concatenate((np.load(file1), np.load(file2)[10:], np.load(file3)[15:]), axis=0)

    if verbose:
        print_list_shape(running_p_array)

    running_p_array = running_p_array[:,:,-take_last:,:,:]

    amount_n = len(running_p_array)
    informations = np.zeros(amount_n)

    if use_BA_information is None:
        use_BA_information = use_BA

    # Loop over rows and compute
    for i in range(amount_n):
        running_p = running_p_array[i][0][0]
        n = i + 1
        if use_BA_information:
            informations[i] = calc_multinomial_BA_input_information(running_p, n)
        else:
            informations[i] = calc_equal_input_multinomial_information(running_p, n)

    return informations


def load_and_calc_noised_information_list(alpha_vec, p_file, w_file, n, take=1):
    from tqdm import tqdm

    running_p_list = load_metrics(p_file)
    running_w_list = load_metrics(w_file)

    amount_d = len(running_p_list)
    reps = len(running_p_list[0])
    epochs = len(running_p_list[0][0])

    output_shape = (amount_d, reps, epochs)  # Shape of the output after reducing the last axis
    informations = np.zeros(output_shape)
    print(f"{len(alpha_vec) =}, {output_shape[0] =}")
    assert len(alpha_vec) == output_shape[0]

    for i in range(output_shape[0]):  # alpha
        a = alpha_vec[i]
        for j in range(output_shape[1]):  # repeats
            for r in tqdm(range(take)):
                x = running_p_list[i][j][-(r + 1)]
                w = running_w_list[i][j][-(r + 1)]

                noise_matrix = get_alpha_matrix(n, a)
                informations[i, j, r] = calc_noised_multinomial_information_with_r(x, w, n, noise_matrix)

    return informations


def calc_take_information(running_p_list, running_w_list, n, take=50):
    from tqdm import tqdm
    amount_d = len(running_p_list)
    reps = len(running_p_list[0])
    epochs = len(running_p_list[0][0])
    dimension = len(running_p_list[0][0][0])

    output_shape = (amount_d, reps, epochs)  # Shape of the output after reducing the last axis
    informations = np.zeros(output_shape)

    for i in tqdm(range(output_shape[0])):  # d
        for j in range(output_shape[1]):  # repeats
            for r in range(take):
                x = running_p_list[i][j][-(r + 1)]
                w = running_w_list[i][j][-(r + 1)]
                informations[i, j, r] = calc_multinomial_information(x, w, n)

    informations = np.max(informations, axis=(1, 2))  # take best from repeats
    return informations


def load_6810():
    test_iter = 6
    running_p_list10 = load_metrics(f'results/running_p_multi_BA_new_next{test_iter}.pkl')
    running_w_list10 = load_metrics(f'results/running_w_multi_BA_new_next{test_iter}.pkl')
    running_p_list10.extend(load_metrics(f'results/running_p_multi_BA_new_next{test_iter-1}.pkl'))
    running_w_list10.extend(load_metrics(f'results/running_w_multi_BA_new_next{test_iter-1}.pkl'))
    training_metrics10 = load_metrics(f'results/training_metrics_multi_BA_3d_6clean_next{test_iter}_2_0.pkl')
    print(f"{training_metrics10['n_trials'] = }")
    print_list_shape(running_p_list10)

    test_iter = 7
    running_p_list6 = load_metrics(f'results/running_p_multi_BA_new_next{test_iter}.pkl')
    running_w_list6 = load_metrics(f'results/running_w_multi_BA_new_next{test_iter}.pkl')
    training_metrics6 = load_metrics(f'results/training_metrics_multi_BA_3d_6clean_next{test_iter}_2_0.pkl')
    print(f"{training_metrics6['n_trials'] = }")
    print_list_shape(running_p_list6)

    test_iter = 8
    running_p_list8 = load_metrics(f'results/running_p_multi_BA_new_next{test_iter}.pkl')
    running_w_list8 = load_metrics(f'results/running_w_multi_BA_new_next{test_iter}.pkl')
    training_metrics8 = load_metrics(f'results/training_metrics_multi_BA_3d_6clean_next{test_iter}_2_0.pkl')
    print(f"{training_metrics8['n_trials'] = }")
    print_list_shape(running_p_list8)

    return running_p_list6, running_w_list6, running_p_list8, running_w_list8, running_p_list10, running_w_list10


def load_6810_new():

    test_number = 3
    date = 3003
    running_p_file = f"results/running_p_{date}_{test_number}.pkl"
    running_w_file = f"results/running_w_{date}_{test_number}.pkl"
    running_p_list6 = load_metrics(running_p_file)
    running_w_list6 = load_metrics(running_w_file)

    test_number = 14
    date = 2903
    running_p_file = f"results/running_p_{date}_{test_number}.pkl"
    running_w_file = f"results/running_w_{date}_{test_number}.pkl"
    running_p_list6.extend(load_metrics(running_p_file))
    running_w_list6.extend(load_metrics(running_w_file))
    print_list_shape(running_p_list6)

    test_number = 2
    date = 3003
    running_p_file = f"results/running_p_{date}_{test_number}.pkl"
    running_w_file = f"results/running_w_{date}_{test_number}.pkl"
    running_p_list8 = load_metrics(running_p_file)
    running_w_list8 = load_metrics(running_w_file)

    test_number = 15
    date = 2903
    running_p_file = f"results/running_p_{date}_{test_number}.pkl"
    running_w_file = f"results/running_w_{date}_{test_number}.pkl"
    running_p_list8.extend(load_metrics(running_p_file))
    running_w_list8.extend(load_metrics(running_w_file))
    print_list_shape(running_p_list8)

    test_number = 1
    date = 3003
    running_p_file = f"results/running_p_{date}_{test_number}.pkl"
    running_w_file = f"results/running_w_{date}_{test_number}.pkl"
    running_p_list10 = load_metrics(running_p_file)
    running_w_list10 = load_metrics(running_w_file)

    test_number = 16
    date = 2903
    running_p_file = f"results/running_p_{date}_{test_number}.pkl"
    running_w_file = f"results/running_w_{date}_{test_number}.pkl"
    running_p_list10.extend(load_metrics(running_p_file))
    running_w_list10.extend(load_metrics(running_w_file))
    print_list_shape(running_p_list10)

    return running_p_list6, running_w_list6, running_p_list8, running_w_list8, running_p_list10, running_w_list10


def load_p_and_w_4d_6(verbose=False, p_file=None, w_file=None):

    if (p_file is not None) and (w_file is not None):
        running_p_array = np.load(p_file)
        running_w_array = np.load(w_file)
    else:
        running_p_array1 = np.load('results/running_p_multi_BA_4d_5_reps.npy')
        running_w_array1 = np.load('results/running_w_multi_BA_4d_5_reps.npy')
        running_p_array2 = np.load('results/running_p_multi_BA_4d_5_reps1.npy')
        running_w_array2 = np.load('results/running_w_multi_BA_4d_5_reps1.npy')
        running_p_array = np.concatenate((running_p_array1, running_p_array2), axis=1)
        running_w_array = np.concatenate((running_w_array1, running_w_array2), axis=1)

    take = 10
    output_shape = running_w_array.shape[:-2]  # Shape of the output after reducing the last axis
    informations = np.zeros((output_shape[0], output_shape[1], take))
    for i in range(output_shape[0]):  # n - trials
        for j in range(output_shape[1]):  # repeats
            for r in range(take):
                x = running_p_array[i, j, -(r+1), :, :]
                w = running_w_array[i, j, -(r+1), :]
                n = i+1
                informations[i, j, r] = calc_multinomial_information(x, w, n)

    if verbose:
        for i in range(output_shape[0]):  # n - trials
            print(f"{i = }")
            arr = informations[i]
            # Get the indices of the maximum value
            index = np.unravel_index(np.argmax(arr), arr.shape)

            print("running_p:")
            formatted_data = np.array2string(running_p_array[i][index[0]][-(index[1]+1)], formatter={'float_kind': lambda x: f"{x:.3f}"})
            print(formatted_data)
            print("running_w:")
            formatted_data = np.array2string(running_w_array[i][index[0]][-(index[1]+1)], formatter={'float_kind': lambda x: f"{x:.3f}"})
            print(formatted_data)

    informations = np.max(informations, axis=(1,2))  # take best from repeats
    return informations


# noise
def calculate_transition_probability(x, noise_matrix, n_trials, count_vectors):
    assert n_trials == np.sum(x)

    transition_prob = np.zeros(len(count_vectors))

    i_jumps = []
    i_p = []
    if x[0] > 0:
        rv = multinomial(x[0], noise_matrix[0])
        for i in range(x[0]+1):
            for ii in range(x[0]-i+1):
                after = [i, ii, x[0]-i-ii]
                p = rv.pmf(after)
                if p>0:
                    i_p.append(p)
                    jump = after
                    jump[0] -= x[0]
                    i_jumps.append(np.array(jump))
                # print(f"{after = }, {p = }")
    else:
        i_jumps = [np.array([0, 0, 0])]
        i_p = [1]

    j_jumps = []
    j_p = []
    if x[1] > 0:
        rv = multinomial(x[1], noise_matrix[1])
        for j in range(x[1] + 1):
            for jj in range(x[1] - j + 1):
                after = [jj, j, x[1] - j - jj]
                p = rv.pmf(after)
                if p > 0:
                    j_p.append(p)
                    jump = after
                    jump[1] -= x[1]
                    j_jumps.append(np.array(jump))
                # print(f"{after = }, {p = }")
    else:
        j_jumps = [np.array([0, 0, 0])]
        j_p = [1]

    k_jumps = []
    k_p = []
    if x[2] > 0:
        rv = multinomial(x[2], noise_matrix[2])
        for k in range(x[2]+1):
            for kk in range(x[2]-k+1):
                after = [kk, x[2]-k-kk, k]
                p = rv.pmf(after)
                if p>0:
                    k_p.append(p)
                    jump = after
                    jump[2] -= x[2]
                    k_jumps.append(np.array(jump))
                # print(f"{after = }, {p = }")
    else:
        k_jumps = [np.array([0, 0, 0])]
        k_p = [1]

    for i, i_jump in enumerate(i_jumps):
        for j, j_jump in enumerate(j_jumps):
            for k, k_jump in enumerate(k_jumps):
                total_jump = i_jump + j_jump + k_jump
                target = x + total_jump
                # Find the index using np.where
                index = np.where(np.all(count_vectors == target, axis=1))[0][0]
                transition_prob[index] += i_p[i]*j_p[j]*k_p[k]
    return transition_prob


def get_alpha_matrix(n=10, alpha=0.25):
    latent_dim = 3
    pyx = [[1, 0, 0],
           [0, 1, 0],
           [alpha, alpha, 1 - 2 * alpha]]

    from itertools import product
    count_vectors_list = [list(count) for count in product(range(n + 1), repeat=latent_dim) if sum(count) == n]
    probability_matrix = []
    for combi in count_vectors_list:
        probability_matrix.append(calculate_transition_probability(combi, pyx, n, count_vectors_list))
    return probability_matrix


def calc_noised_multinomial_information(x, n, noise_matrix):
    p = calc_multinomial_channel(x, n)
    noised_channel = np.matmul(np.asarray(p), noise_matrix)
    C, _ = blahut_arimoto(noised_channel)
    return C


def calc_noised_multinomial_information_with_r(x, r, n, noise_matrix):
    assert len(x) == len(r)
    p_y_x = calc_multinomial_channel(x, n)
    noised_channel = np.matmul(np.asarray(p_y_x), noise_matrix)
    C = calc_information(np.asarray(noised_channel), r)
    return C


def calc_pyk_noised(x, n, noise_matrix):
    p = calc_multinomial_channel(x, n)
    noised_channel = np.matmul(np.asarray(p), noise_matrix)
    I, r = blahut_arimoto(noised_channel)
    p_y_k = np.matmul(np.array([r]), noised_channel)
    return p_y_k, I, r


def calc_noised_multinomial_BA(x, n, noise_matrix):
    p = calc_multinomial_channel(x, n)
    noised_channel = np.matmul(np.asarray(p), noise_matrix)
    C, r = blahut_arimoto(noised_channel)
    return C, r


# Deep
def probs_to_logits(probs):
    # Ensure that the probabilities are valid (no zeroes, as log(0) is undefined)
    probs = torch.clamp(probs, min=1e-16)

    # Compute logits from probs
    logits = torch.log(probs)

    return logits


def get_p_from_explicit_model(model, condition_dim, device, multi=False):
    batch = torch.eye(condition_dim)
    batch = batch.to(device)
    if model.numerical_input:
        # Convert one-hot encoded input to numerical indices
        num_categories = batch.size(-1)  # Number of categories (size of one-hot vector)
        batch = batch.argmax(dim=-1, keepdim=True).float()  # Indices as float
        batch = batch / (num_categories - 1)  # Normalize to [0, 1]
    mean_p = model.encoder(batch)
    if not multi:
        mean_p = mean_p[:, 0]
    return mean_p.detach().cpu().numpy()


def get_multi_encoder(model, condition_dim, device="cpu"):
    from model_classes import ModifiedEncoder
    # Assuming `trained_model` is your already trained model
    modified_model = ModifiedEncoder(model)

    model.eval()

    with torch.no_grad():
        input_data = torch.eye(condition_dim).to(device)
        probs = modified_model(input_data, return_pre_activation=False).cpu()
    return probs
