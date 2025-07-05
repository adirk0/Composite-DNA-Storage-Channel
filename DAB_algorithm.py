import time
import numpy as np
import scipy.special
from scipy.optimize import fmin
from scipy.optimize import Bounds, shgo
from utils import kl_divergence
from blahut_arimoto import blahut_arimoto


def binomial_p_with_error(n, x, y, error=0):
    new_x = x*(1-error)+(1-x)*error
    return scipy.special.binom(n, y)*(new_x**y)*((1-new_x)**(n-y))


def calc_binomial_channel(x, n, error=0):
    y = np.array(range(n+1))
    p_y_x = np.asarray([binomial_p_with_error(n, i, y, error=error) for i in x])
    return p_y_x


def f(n, x, p_y_k, error=0):
    # binomial point_kl
    p = calc_binomial_channel(x, n, error=error)
    return kl_divergence(p, p_y_k)


def calc_kl_diff(delta, closest_x, max_x, start, middle, end, n, error=0):
    x1 = closest_x + delta * (max_x - closest_x)
    x2 = 1 - x1
    x = start + [x1] + middle + [x2] + end
    p = calc_binomial_channel(x, n, error=error)
    I, r = blahut_arimoto(np.asarray(p))
    p_y_k = np.matmul(np.array([r]), p)
    return kl_divergence(p, p_y_k)


def grad_DAB_step_error(new_x, n, init_threshold=1e-4, init_delta=1e-3, error=0):
    delta = init_delta
    threshold = init_threshold
    max_iter = 2000

    iter_num = 0
    t_start = time.time()

    # optimize the allocation of probability to the current mass point locations
    while True:
        if iter_num == 500:
            print("warning")
        assert iter_num < max_iter

        x = new_x

        p = calc_binomial_channel(x, n, error=error)
        I, r = blahut_arimoto(np.asarray(p))
        p_y_k = np.matmul(np.array([r]), p)

        D = 0
        max_x = -1

        bounds = Bounds([0], [1])
        res = shgo(lambda x: -f(n, x, p_y_k, error=error), bounds, sampling_method='sobol')

        if -res.fun > D:
            D = -res.fun
            max_x = res.x

        # take the first maximum
        if max_x > 0.5:
            max_x = 1 - max_x

        assert I < D + 1e-1

        print("n:iter = %02d :" % n, "%03d" % iter_num, ", delta = %.4f" % delta,
              ", I  = %.7f" % I, ", max_D =  %.7f" % D, "diff: %+.2e" % (D - I),
              "finish: %d" % (D - I <= threshold),
              "max_x = %50s" % max_x, ", x after: = ", x)

        if D - I <= threshold:
            num_point = len(x)
            t_end = time.time()
            total_time = t_end - t_start

            return I, r, x, iter_num,  num_point, total_time

        iter_num += 1

        x_array = np.array(x, dtype=object)
        x_flat = np.array([float(xi) if isinstance(xi, (int, float, np.number)) else float(xi[0]) for xi in x_array])
        idx = (np.abs(x_flat - max_x)).argmin()

        closest_x = x[idx]
        x_size = np.shape(x_array)[0]

        if np.abs(closest_x - max_x) >= np.abs(0.5 - max_x) and not(x_size % 2):  # even
            half = x_size // 2
            new_x = x[:half] + [0.5] + x[half:]

        else:
            if np.abs(closest_x - max_x) >= np.abs(0.5 - max_x):  # odd
                assert closest_x == 0.5
                x1 = 0.5 + init_delta * (max_x - 0.5)
                x2 = 1 - x1
                # new_x = x[:idx] + [x1, x2] + x[idx+1:]

                start = x[:idx]
                end = x[idx+1:]
                middle = []
            else:
                middle = x[idx+1: (x_size - 1 - idx)]

                start = x[:idx]
                end = x[(x_size - idx):]

            maximum = fmin(lambda d: -calc_kl_diff(d, closest_x, max_x, start, middle, end, n, error=error),
                           0, full_output=1, disp=0)
            delta = maximum[0]

            if delta < 2e-5:
                print("delta: ", delta)
            x1 = closest_x + delta*(max_x-closest_x)
            x2 = 1-x1
            new_x = start + [x1] + middle + [x2] + end
