import numpy as np
from scipy.special import binom
from blahut_arimoto import blahut_arimoto
from utils import calc_information


# Load Information
def load_support_information(use_BA=False, file=None, take_last=1, use_BA_information=None):

    if file is None:
        if use_BA:
            running_p_array_20 = np.load('results/real_running_p_array_20_BA.npy')
            running_p_array_30 = np.load('results/real_running_p_array_30_BA.npy')
            running_p_array = np.load('results/real_running_p_array_50_BA.npy')
            running_p_array[:20] = running_p_array_20
            running_p_array[20:30] = running_p_array_30[20:30]

        else:
            running_p_array_20 = np.load('results/running_p_array_20.npy')
            running_p_array = np.load('results/running_p_array_50.npy')
            running_p_array[:20] = running_p_array_20
    else:
        running_p_array = np.load(file)

    running_p_array = np.sort(running_p_array, axis=-1)[:, :, -take_last:, :]

    output_shape = running_p_array.shape[:-1]  # Shape of the output after reducing the last axis
    informations = np.zeros(output_shape)

    n_array = [i for i in range(1,51)]
    if use_BA_information is None:
        use_BA_information = use_BA

    # Loop over rows and compute
    for i in range(running_p_array.shape[0]):

        if use_BA_information:
            informations[i] = np.apply_along_axis(calc_BA_input_information, axis=-1, arr=running_p_array[i], n=n_array[i])
        else:
            informations[i] = np.apply_along_axis(calc_equal_input_information, axis=-1, arr=running_p_array[i], n=n_array[i])

    informations = np.max(informations, axis=(1,2))
    return informations


#  Binomial
def binomial_p(n, x, y):
    return binom(n, y)*(x**y)*((1-x)**(n-y))


def calc_binomial_channel(x, n):
    y = np.array(range(n+1))
    p_y_x = np.asarray([binomial_p(n, i, y) for i in x])
    return p_y_x


def calc_binomial_BA(x, n):
    p = calc_binomial_channel(x, n)
    C, r = blahut_arimoto(np.asarray(p))
    return C, r


def calc_BA_input_information(x, n):
    C, _ = calc_binomial_BA(x, n)
    return C


def calc_binomial_information(x, r, n):
    assert len(x) == len(r)
    p_y_x = calc_binomial_channel(x, n)
    C = calc_information(p_y_x, r)
    return C


def calc_equal_input_information(x, n):
    m = len(x)
    r = np.array([1/m]*m)
    return calc_binomial_information(x, r, n)


def inv_square_kernel(x):
    d = x - 0.5
    x = np.sign(d)*np.sqrt(np.abs(d)/2)+0.5
    return x


def fixed_support_information(support, n_vec, use_BA=False):
    if use_BA:
        return [calc_BA_input_information(support, n) for n in n_vec]
    return [calc_equal_input_information(support, n) for n in n_vec]

