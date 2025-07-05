import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize, Bounds
from visualize import plot_simplex_2d_without_slider, plot_simplex_2d_ax
from utils import calc_multinomial_BA_input_information, calc_multinomial_BA,\
    calculate_transition_probability, calc_noised_multinomial_information
from sys import float_info

eps = float_info.epsilon


def solve(n, k, d, verbose=False):
    # Define the cost function
    def cost_fun(flat_x):
        # Reshape flat_x into a list of d arrays of size k
        x = [flat_x[i * k:(i + 1) * k] for i in range(d)]
        return -calc_multinomial_BA_input_information(x, n)

    # Define the bounds: x_i >= 0 for all i and x_i <= 1
    bounds = [(0, 1) for _ in range(d * k)]

    # Define the constraints: each group of k elements sums to 1
    linear_constraints = [
        {
            'type': 'eq',
            'fun': lambda x, i=i: np.sum(x[i * k:(i + 1) * k]) - 1
        }
        for i in range(d)
    ]

    # Generate different random probability vectors for initialization
    def random_probability_vector(size):
        vec = np.random.rand(size)
        return vec / np.sum(vec)

    x0 = np.concatenate([random_probability_vector(k) for _ in range(d)])

    # Minimize the cost function
    res = minimize(cost_fun, x0, constraints=linear_constraints, bounds=bounds)
    if verbose:
        formatted_data = np.array2string(res.x, formatter={'float_kind': lambda x: f"{x:.2f}"})
        print(formatted_data)
    return -res.fun


def solve6corners(n, alpha, verbose=True):
    latent_dim = 3
    pyx = [[1, 0, 0],
           [0, 1, 0],
           [alpha, alpha, 1 - 2 * alpha]]

    from itertools import product
    count_vectors_list = [list(count) for count in product(range(n + 1), repeat=latent_dim) if sum(count) == n]
    probability_matrix = []
    for combi in count_vectors_list:
        probability_matrix.append(calculate_transition_probability(combi, pyx, n, count_vectors_list))

    def cost_fun(a):
        a = float(a)
        x = [np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1]),
             np.array([1 / 2, 1 / 2, 0]),  np.array([a, 0, 1-a]), np.array([0, a, 1-a])]
        return - calc_noised_multinomial_information(x, n, probability_matrix)

    bounds = Bounds(0, 1)
    x0 = 0.3

    res = minimize(cost_fun, x0, bounds=bounds)
    if verbose:
        print(res.x)
    return -res.fun


def solve6edge(n, alpha=0.25, verbose=True):
    latent_dim = 3
    pyx = [[1, 0, 0],
           [0, 1, 0],
           [alpha, alpha, 1 - 2 * alpha]]

    from itertools import product
    count_vectors_list = [list(count) for count in product(range(n + 1), repeat=latent_dim) if sum(count) == n]
    probability_matrix = []
    for combi in count_vectors_list:
        probability_matrix.append(calculate_transition_probability(combi, pyx, n, count_vectors_list))

    def cost_fun(a):
        a = float(a)
        x = [np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1]),  np.array([a, 1-a-eps, +eps]), np.array([1-a, a, 0])]
        return - calc_noised_multinomial_information(x, n, probability_matrix)

    bounds = Bounds(0, 1)
    x0 = 0.45

    res = minimize(cost_fun, x0, bounds=bounds)
    if verbose:
        print(res.x)
    return -res.fun


def solve6middle(n, alpha=0.25, verbose=True, simplex=False):
    latent_dim = 3
    pyx = [[1, 0, 0],
           [0, 1, 0],
           [alpha, alpha, 1 - 2 * alpha]]

    from itertools import product
    count_vectors_list = [list(count) for count in product(range(n + 1), repeat=latent_dim) if sum(count) == n]
    probability_matrix = []
    for combi in count_vectors_list:
        probability_matrix.append(calculate_transition_probability(combi, pyx, n, count_vectors_list))

    def cost_fun(c):
        a = float(c[0])
        b = float(c[1])
        # note i use also "np.array([1 / 2, 1 / 2 - eps, eps])" as BA arimoto not support with P(y_i) = 0
        # As well to avoid the case a = 0
        x = [np.array([1, 0, 0]), np.array([0, 1, 0]),
             np.array([1-b-2*eps , b+eps, eps]), np.array([b, 1-b - eps, eps]),  np.array([b, 1-b , 0]),
             np.array([a, 0, 1-a]), np.array([0, a+eps, 1-a-eps])]
        return - calc_noised_multinomial_information(x, n, probability_matrix)

    bounds = Bounds(0, 1)
    x0 = [0.4, 0.4]

    res = minimize(cost_fun, x0, bounds=bounds)
    if verbose:
        print(res.x)
        if simplex:
            from visualize import test_plot_simplex_inter_result
            a = res.x[0]
            b = res.x[1]
            support = [np.array([1, 0, 0]), np.array([0, 1, 0]),
                 np.array([1-b-2*eps , b+eps, eps]), np.array([b, 1-b - eps, eps]),  np.array([b, 1-b , 0]),
                 np.array([a, 0, 1-a]), np.array([0, a+eps, 1-a-eps])]
            test_plot_simplex_inter_result(support, n=10, block=False)
    return -res.fun


def solve5(n, verbose=False):
    eps = float_info.epsilon

    def cost_fun(a):
        a = float(a)
        x = [np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([1 / 2, 1 / 2, 0]),  np.array([a, 0, 1-a]), np.array([0, a, 1-a]), np.array([1 / 2, 1 / 2 - eps, eps])]
        return - calc_multinomial_BA_input_information(x, n)

    bounds = Bounds(0, 1)
    x0 = 0.5

    res = minimize(cost_fun, x0, bounds=bounds)
    if verbose:
        print(res.x)
    return -res.fun


def plot_local_minimas(usetex=True, verify_shift=False, verify_middle_shift=False, block=False):
    # solve_10d6n_configuration
    # Figure 3: Mutual information achieved by different constellations for C_{n=6,k=3}.

    n = 6

    fig = plt.figure(figsize=(12, 6))
    plt.rcParams['text.usetex'] = usetex
    plt.rcParams["font.style"] = 'italic'
    plt.rcParams['font.family'] = 'serif'

    x = [np.array([1, 0, 0]), np.array([0, 0, 1]), np.array([0, 1, 0]),
                       np.array([0, 1 / 2, 1 / 2]), np.array([1 / 2, 0, 1 / 2]), np.array([1 / 2, 1 / 2, 0])]
    c, r = calc_multinomial_BA(x, n)
    print(f"{c = }, {r = }")
    plot_simplex_2d_ax(x, r=r, c=c, block=False, ax=fig.add_subplot(2, 4, 1))

    x = [np.array([1, 0, 0]), np.array([0, 0, 1]), np.array([0, 1, 0]),
                       np.array([0, 1 / 2, 1 / 2]), np.array([1 / 2, 0, 1 / 2]), np.array([1 / 2, 1 / 2, 0]),
                       np.array([1 / 3, 1 / 3, 1 / 3])]
    c, r = calc_multinomial_BA(x, n)
    print(f"{c = }, {r = }")
    plot_simplex_2d_ax(x, r=r, c=c, block=False, ax=fig.add_subplot(2, 4, 5))

    if verify_shift:

        def solve_shift(n, verbose=False, add_middle=False):
            def cost_fun(a):
                a = float(a)

                x = [np.array([1, 0, 0]), np.array([0, 0, 1]), np.array([0, 1, 0]),
                     np.array([0, 1 / 2+a, 1 / 2-a]), np.array([1 / 2-a, 0, 1 / 2+a]), np.array([1 / 2+a, 1 / 2-a, 0])]
                if add_middle:
                    x.append(np.array([1 / 3, 1 / 3, 1 / 3]))
                return - calc_multinomial_BA_input_information(x, n)

            bounds = Bounds(0, 0.5)
            x0 = 0.45

            res = minimize(cost_fun, x0, bounds=bounds)
            if verbose:
                print(res)
                print(res.x)
            return -res.fun

        solve_shift(n, verbose=True)
        solve_shift(n, verbose=True, add_middle=True)

        a = 0.1
        x = [np.array([1, 0, 0]), np.array([0, 0, 1]), np.array([0, 1, 0]),
             np.array([0, 1 / 2 + a, 1 / 2 - a]), np.array([1 / 2 - a, 0, 1 / 2 + a]), np.array([1 / 2 + a, 1 / 2 - a, 0])]
        c, r = calc_multinomial_BA(x, n)
        print(f"{c = }, {r = }")
        plot_simplex_2d_without_slider(x, r=r, block=False)

    def solve1edge(n, verbose=True, add_middle=False):

        def cost_fun(a):
            a = float(a)
            x = [np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1]),
                 np.array([0, 1 / 2, 1 / 2]), np.array([1 / 2, 0, 1 / 2]),
                 np.array([a, 1 - a, 0]), np.array([1 - a, a, 0])]
            if add_middle:
                x.append(np.array([1 / 3, 1 / 3, 1 / 3]))
            return - calc_multinomial_BA_input_information(x, n)

        bounds = Bounds(0, 1)
        x0 = 0.45

        res = minimize(cost_fun, x0, bounds=bounds)
        if verbose:
            print(res)
            print(res.x)

        a = res.x[0]
        support = [np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1]),
                   np.array([0, 1 / 2, 1 / 2]), np.array([1 / 2, 0, 1 / 2]),
                   np.array([a, 1 - a, 0]), np.array([1 - a, a, 0])]
        if add_middle:
            support.append(np.array([1 / 3, 1 / 3, 1 / 3]))

        return support, -res.fun

    x, c_solve = solve1edge(n)
    c, r = calc_multinomial_BA(x, n)
    print(f"{c = }, {c_solve = }, {r = }")
    plot_simplex_2d_ax(x, r=r, c=c, block=False, ax=fig.add_subplot(2, 4, 2))

    x, c_solve = solve1edge(n, add_middle=True)
    c, r = calc_multinomial_BA(x, n)
    print(f"{c = }, {c_solve = }, {r = }")
    plot_simplex_2d_ax(x, r=r, c=c, block=False, ax=fig.add_subplot(2, 4, 6))

    if verify_middle_shift:

        def solve1edge1middle(n, verbose=True):

            def cost_fun(c):

                a = float(c[0])
                b = float(c[1])
                x = [np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1]),
                     np.array([0, 1 / 2, 1 / 2]), np.array([1 / 2, 0, 1 / 2]),
                     np.array([a, 1 - a, 0]), np.array([1 - a, a, 0]),
                     np.array([1 / 3-b, 1 / 3-b, 1 / 3+2*b])]
                return - calc_multinomial_BA_input_information(x, n)

            bounds = Bounds(-0.2, 0.5)
            x0 = [0.45, 0.05]

            res = minimize(cost_fun, x0, bounds=bounds)
            if verbose:
                print(res)
                print(res.x)

            a = res.x[0]
            b = res.x[1]
            support = [np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1]),
                       np.array([0, 1 / 2, 1 / 2]), np.array([1 / 2, 0, 1 / 2]),
                       np.array([a, 1 - a, 0]), np.array([1 - a, a, 0]),
                       np.array([1 / 3-b, 1 / 3-b, 1 / 3+2*b])]

            return support, -res.fun

        x, c_solve = solve1edge1middle(n)
        c, r = calc_multinomial_BA(x, n)
        print(f"{c = }, {c_solve = }, {r = }")
        plot_simplex_2d_without_slider(x, r=r, c=c, block=False)

        def solve_shifted1edge1middle(n, verbose=True):

            def cost_fun(d):

                a = float(d[0])
                b = float(d[1])
                c = float(d[2])
                x = [np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1]),
                     np.array([0, 1 / 2-c, 1 / 2+c]), np.array([1 / 2-c, 0, 1 / 2+c]),
                     np.array([a, 1 - a, 0]), np.array([1 - a, a, 0]),
                     np.array([1 / 3-b, 1 / 3-b, 1 / 3+2*b])]
                return - calc_multinomial_BA_input_information(x, n)

            bounds = Bounds(-0.1, 0.5)
            x0 = [0.45, 0.05, 0.05]

            res = minimize(cost_fun, x0, bounds=bounds)
            if verbose:
                print(res)
                print(res.x)

            a = res.x[0]
            b = res.x[1]
            c = res.x[2]
            support = [np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1]),
                       np.array([0, 1 / 2-c, 1 / 2+c]), np.array([1 / 2-c, 0, 1 / 2+c]),
                       np.array([a, 1 - a, 0]), np.array([1 - a, a, 0]),
                       np.array([1 / 3-b, 1 / 3-b, 1 / 3+2*b])]

            return support, -res.fun

        x, c_solve = solve_shifted1edge1middle(n)
        c, r = calc_multinomial_BA(x, n)
        print(f"{c = }, {c_solve = }, {r = }")
        plot_simplex_2d_without_slider(x, r=r, c=c, block=False)

    def solve2edge(n, verbose=True, add_middle=False):

        def cost_fun(a):
            a = float(a)
            x = [np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1]),
                 np.array([0, 1 / 2, 1 / 2]),
                 np.array([a, 0, 1 - a]), np.array([1 - a, 0, a]),
                 np.array([a, 1 - a, 0]), np.array([1 - a, a, 0])]
            if add_middle:
                x.append(np.array([1 / 3, 1 / 3, 1 / 3]))
            return - calc_multinomial_BA_input_information(x, n)

        bounds = Bounds(0, 1)
        x0 = 0.45

        res = minimize(cost_fun, x0, bounds=bounds)
        if verbose:
            print(res)
            print(res.x)

        a = res.x[0]
        support = [np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1]),
                   np.array([0, 1 / 2, 1 / 2]),
                   np.array([a, 0, 1 - a]), np.array([1 - a, 0, a]),
                   np.array([a, 1 - a, 0]), np.array([1 - a, a, 0])]
        if add_middle:
            support.append(np.array([1 / 3, 1 / 3, 1 / 3]))

        return support, -res.fun

    x, c_solve = solve2edge(n)
    c, r = calc_multinomial_BA(x, n)
    print(f"{c = }, {c_solve = }, {r = }")
    plot_simplex_2d_ax(x, r=r, c=c, block=False, ax=fig.add_subplot(2, 4, 3))

    x, c_solve = solve2edge(n, add_middle=True)
    c, r = calc_multinomial_BA(x, n)
    print(f"{c = }, {c_solve = }, {r = }")
    plot_simplex_2d_ax(x, r=r, c=c, block=False, ax=fig.add_subplot(2, 4, 7))

    def solve3edge(n, verbose=True, add_middle=False):

        def cost_fun(a):
            a = float(a)
            x = [np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1]),
                 np.array([0, a, 1 - a]), np.array([0, 1 - a, a]),
                 np.array([a, 0, 1 - a]), np.array([1 - a, 0, a]),
                 np.array([a, 1 - a, 0]), np.array([1 - a, a, 0])]
            if add_middle:
                x.append(np.array([1 / 3, 1 / 3, 1 / 3]))
            return - calc_multinomial_BA_input_information(x, n)

        bounds = Bounds(0, 1)
        x0 = 0.45

        res = minimize(cost_fun, x0, bounds=bounds)
        if verbose:
            print(res)
            print(res.x)

        a = res.x[0]
        support = [np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1]),
                   np.array([0, a, 1 - a]), np.array([0, 1 - a, a]),
                   np.array([a, 0, 1 - a]), np.array([1 - a, 0, a]),
                   np.array([a, 1 - a, 0]), np.array([1 - a, a, 0])]
        if add_middle:
            support.append(np.array([1 / 3, 1 / 3, 1 / 3]))

        return support, -res.fun

    x, c_solve = solve3edge(n)
    c, r = calc_multinomial_BA(x, n)
    print(f"{c = }, {c_solve = }, {r = }")
    plot_simplex_2d_ax(x, r=r, c=c, block=False, ax=fig.add_subplot(2, 4, 4))

    x, c_solve = solve3edge(n, add_middle=True)
    c, r = calc_multinomial_BA(x, n)
    print(f"{c = }, {c_solve = }, {r = }")
    plot_simplex_2d_ax(x, r=r, c=c, block=False, ax=fig.add_subplot(2, 4, 8))

    plt.show(block=block)
