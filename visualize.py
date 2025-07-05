import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as lines
from matplotlib.gridspec import GridSpec
from matplotlib.colors import Normalize
from blahut_arimoto import blahut_arimoto
from DAB_algorithm import grad_DAB_step_error
from MDAB_algorithm import multi_grad_DAB_step
from utils import restore_file, plot_from_fig, return_corners, return_2_corners, calc_pyk, point_kl, \
    calc_full_multinomial_channel, fixed_multinomial_support_information, \
    load_support_information_multinomial, load_and_calc_noised_information_list, get_alpha_matrix, calc_pyk_noised, \
    calc_take_information, load_metrics, load_6810, load_6810_new, load_p_and_w_4d_6
from discretization import load_support_information, fixed_support_information, inv_square_kernel
from sys import float_info


def map_colors(p3dc, func, cmap='viridis', cmin=0.0, cmax=1.0):
    """
    Color a tri-mesh according to a function evaluated in each barycentre.

    p3dc: a Poly3DCollection, as returned e.g. by ax.plot_trisurf
    func: a single-valued function of 3 arrays: x, y, z
    cmap: a colormap NAME, as a string

    Returns a ScalarMappable that can be used to instantiate a colorbar.
    """

    from matplotlib.cm import ScalarMappable, get_cmap
    from matplotlib.colors import Normalize, LinearSegmentedColormap
    from numpy import array

    # reconstruct the triangles from internal data
    x, y, z, _ = p3dc._vec
    slices = p3dc._segslices
    triangles = array([array((x[s], y[s], z[s])).T for s in slices])

    # compute the barycentres for each triangle
    xb, yb, zb = triangles.mean(axis=1).T

    # compute the function in the barycentres
    values = func(xb, yb, zb)

    # usual stuff
    norm = Normalize()
    # colors = get_cmap(cmap)(norm(values))

    # Get the colormap and extract a sub-range
    original_cmap = get_cmap(cmap)
    new_cmap = LinearSegmentedColormap.from_list(
        f'{cmap}_sub',
        original_cmap(np.linspace(cmin, cmax, 256))
    )

    # Apply the colormap
    colors = new_cmap(norm(values))

    # set the face colors of the Poly3DCollection
    p3dc.set_fc(colors)

    # if the caller wants a colorbar, they need this
    return ScalarMappable(cmap=cmap, norm=norm)


def plot_approximation_capacity_multidim(max_n, fix_x, label, color='purple'):
    C_vec = []
    for n in range(1, max_n + 1):
        p = calc_full_multinomial_channel(fix_x, n)
        C, r = blahut_arimoto(np.asarray(p))
        C_vec.append(C)
    print("approx C_vec: ", C_vec)
    plt.plot(range(1, max_n + 1), C_vec, color=color, label=str(label))


def compare_capacity_4d(usetex=True, block=False):
    f1 = restore_file('results/multidim.pickle')
    f = plt.figure()

    plt.rcParams['text.usetex'] = usetex
    plt.rcParams['font.family'] = 'serif'

    amount = 5
    color_shift = 2
    cmap = plt.cm.hot

    def used_color(i, cmap=plt.cm.BuPu, amount=4, color_shift=2):
        return cmap((i + color_shift) / (amount + color_shift))

    plot_from_fig(f1, sub=0, label="M-DAB", color=used_color(0, cmap, amount=amount, color_shift=color_shift))
    plt.close(f1)
    max_n = 10

    fix_x = return_corners(dim=4)
    plot_approximation_capacity_multidim(max_n, fix_x, label="Uniform composite", color=used_color(2, cmap, amount=amount, color_shift=color_shift))

    x_range = [a for a in range(1, max_n + 1)]
    plt.plot(x_range, [np.log2(4)]*max_n, '--k', label='Non composite bound')
    plt.plot(x_range, [np.log2(15)]*max_n, 'k',linestyle='dashdot', label='Uniform composite bound')

    plt.xlim([1, 10])
    plt.legend(fontsize="16", bbox_to_anchor=(0.30, -0.05, 0.5, 0.5))
    plt.ylabel("$C_{n,k=4}$", fontsize="18")  # "Capacity"
    plt.xlabel("$n$", fontsize="18")  # Number of Multinomial Trials
    plt.grid()
    plt.show(block=block)


def plot_simplex_maximizer_example(usetex=True, block=False):
    # Figure 1: The first iteration of the M-DAB algorithm for C_{n=7,k=3}.

    n = 7
    new_x = [np.array([1 / 3, 1 / 3, 1 / 3]), np.array([0.68185518, 0.31814482, 0]), np.array([1, 0, 0])]
    max_x = [0.6165029,  0.19174855, 0.19174855]
    p_y_k, I, r = calc_pyk(new_x, n)

    def f_3d_vec(x, y, z):
        return [point_kl(n, [x[i], y[i], z[i]], p_y_k) for i in range(len(x))]

    n_x = 100
    n_y = 100
    xd = np.linspace(0, 1, n_x)
    yd = np.linspace(0, 1, n_y)
    x, y = np.meshgrid(xd, yd)

    x = np.ravel(x)
    y = np.ravel(y)
    xy = list(zip(x, y))
    triangle = list(filter(lambda a: a[0] + a[1] <= 1, xy))
    t_len = len(triangle)
    t_x, t_y = zip(*triangle)
    t_z = [1- t_x[i]-t_y[i] for i in range(t_len)]

    fig = plt.figure()
    plt.rcParams['text.usetex'] = usetex
    plt.rcParams["font.style"] = 'italic'
    plt.rcParams['font.family'] = 'serif'

    ax = plt.axes(projection='3d')
    p3dc = ax.plot_trisurf(t_x, t_y, t_z, alpha=0.5)

    # change the face colors
    mappable = map_colors(p3dc, f_3d_vec, 'YlOrRd')
    ax.view_init(azim=45, elev=20)


    alpha = 50

    # x_0 = x_1
    zline = np.linspace(0, 1, alpha)
    yline = 0.5 - 0.5*zline
    xline = 0.5 - 0.5*zline
    ax.plot3D(xline, yline, zline, 'k', linewidth=2, alpha=1)

    # x_1 = x_2
    xline = np.linspace(0, 1, alpha)
    yline = 0.5 - 0.5*xline
    zline = 0.5 - 0.5*xline
    ax.plot3D(xline, yline, zline, 'k', linewidth=2, alpha=1)

    # x_0 = x_2
    yline = np.linspace(0, 1,  alpha)
    xline = 0.5 - 0.5*yline
    zline = 0.5 - 0.5*yline
    ax.plot3D(xline, yline, zline, 'k', linewidth=2, alpha=1)

    # x_2 = 0
    yline = np.linspace(0, 1,  alpha)
    xline = 1 - yline
    zline = 0*yline
    ax.plot3D(xline, yline, zline, 'k', linewidth=2, alpha=1)

    # x_1 = 0
    xline = np.linspace(0, 1,  alpha)
    zline = 1 - xline
    yline = 0*xline
    ax.plot3D(xline, yline, zline, 'k', linewidth=2, alpha=1)

    # x_0 = 0
    yline = np.linspace(0, 1,  alpha)
    zline = 1 - yline
    xline = 0*yline
    ax.plot3D(xline, yline, zline, 'k', linewidth=2, alpha=1)

    symbols = np.transpose(new_x)
    ax.scatter(symbols[0], symbols[1], symbols[2],
               c='maroon', s=70, marker='o', edgecolor='k', alpha=1, label='Input distribution mass point')
    ax.scatter(max_x[0], max_x[1], max_x[2],
               c='darkorange', s=70, marker='X', edgecolor='k', alpha=1, label='KL divergence maximizer')

    ax.set_xlabel("$x_{1}$", fontsize="18")
    ax.set_ylabel("$x_{2}$", fontsize="18")
    ax.set_zlabel("$x_{3}$", fontsize="18")

    plt.legend(fontsize="16", bbox_to_anchor=(0., 0.4, 0.5, 0.5))
    plt.show(block=block)


def project_3d(symbols):
    def normalized(v):
        return v / np.linalg.norm(v)

    A = np.array([1, 0, 0])
    B = np.array([0, 1, 0])
    C = np.array([0, 0, 1])

    ax1 = normalized(B - A)
    ax2 = normalized(C - (A + B) / 2)

    v1 = np.array([np.array(ax1)])
    v2 = np.array([np.array(ax2)])

    A_rep = np.repeat(np.transpose(np.array([A])), np.shape(symbols)[1], axis=1)
    centered_symbols = symbols - A_rep

    x = np.matmul(v1, centered_symbols)[0]
    y = np.matmul(v2, centered_symbols)[0]

    return x, y


def ret_triangle(res=25, edge_res=None):

    n_x = res
    n_y = res
    xd = np.linspace(0, 1, n_x)
    yd = np.linspace(0, 1, n_y)

    if edge_res is not None:
        n_x_log = edge_res
        n_y_log = edge_res
        y = np.logspace(-10, -1, n_y_log)
        yd = np.concatenate([yd, y], 0)
        x = 1 - np.logspace(-10, -1, n_x_log)
        xd = np.concatenate([xd, x], 0)

    # Meshgrid points and triangle selection
    x, y = np.meshgrid(xd, yd)
    x = np.ravel(x)
    y = np.ravel(y)
    xy = list(zip(x, y))
    triangle = list(filter(lambda a: a[0] + a[1] <= 1, xy))
    t_len = len(triangle)
    t_x, t_y = zip(*triangle)
    t_z = [1- t_x[i]-t_y[i] for i in range(t_len)]

    return t_x, t_y, t_z


def plot_simplex_2d_without_slider(running_model, r=None, c=None,  block=True, inline=True,
                                   background=False, n_test=101, t_color=None, title=None):
    # Lines for projection
    alpha = 50
    lines = [
        (np.zeros(alpha), np.linspace(0, 1, alpha), 1 - np.linspace(0, 1, alpha)),
        (np.linspace(0, 1, alpha), np.zeros(alpha), 1 - np.linspace(0, 1, alpha)),
        (np.linspace(0, 1, alpha), 1 - np.linspace(0, 1, alpha), np.zeros(alpha)),
        (np.linspace(0, 1, alpha), 0.5 - 0.5 * np.linspace(0, 1, alpha), 0.5 - 0.5 * np.linspace(0, 1, alpha)),
        (0.5 - 0.5 * np.linspace(0, 1, alpha), np.linspace(0, 1, alpha), 0.5 - 0.5 * np.linspace(0, 1, alpha)),
        (0.5 - 0.5 * np.linspace(0, 1, alpha), 0.5 - 0.5 * np.linspace(0, 1, alpha), np.linspace(0, 1, alpha))
    ]

    # Project the lines to 2D
    projected_lines = []
    for line in lines:
        line_3d = np.array(line)
        proj_x, proj_y = project_3d(line_3d)
        projected_lines.append((proj_x, proj_y))

    fig = plt.figure(figsize=(4, 4))

    if c is not None:
        plt.title(f"I(X;Y) = {c:.4}")
    plt.xlim([-0.1, 1.5])
    plt.ylim([-0.1, 1.5])
    plt.axis("off")

    # Plot symbols
    num_plots = len(running_model)
    condition_dim = np.shape(running_model)[0]
    if r is None:
        r = np.array([1/condition_dim]*condition_dim)

    norm = Normalize(vmin=0, vmax=1)

    if background:
        t_x, t_y, t_z = ret_triangle(res=n_test)
        # Project the 3D triangle vertices to 2D
        triangle_points = np.array([t_x, t_y, t_z])
        proj_t_x, proj_t_y = project_3d(triangle_points)

        # Normalize the colormap to match the range of t_color[0]
        tricolor = plt.tripcolor(proj_t_x, proj_t_y, t_color[0], shading='gouraud', cmap='YlOrRd', norm=norm, alpha=0.7)

    matplotlib.colors.ColorConverter.colors['l'] = (0.3, 0.3, 0.3)

    # Plot lines
    for line_idx, (proj_x, proj_y) in enumerate(projected_lines):
        linestyle = 'k-'
        if inline and line_idx >= 3:
            linestyle = 'l--'

        plt.plot(proj_x, proj_y, linestyle, linewidth=2, zorder=0)

    symbols = np.array(running_model).T
    proj_symbols_x, proj_symbols_y = project_3d(symbols)

    scatter = plt.scatter(proj_symbols_x, proj_symbols_y, c=1-r,  cmap='YlOrRd', s=70*10*r,
                          marker='o', edgecolor='k', norm=norm, label='Projected Symbols')
    if title is not None:
        plt.title(title)
    plt.show(block=block)


def plot_simplex_2d_ax(running_model, r=None, c=None,  block=True, inline=True, ax=None, usetex=True):
    # Lines for projection
    alpha = 50
    lines = [
        (np.zeros(alpha), np.linspace(0, 1, alpha), 1 - np.linspace(0, 1, alpha)),
        (np.linspace(0, 1, alpha), np.zeros(alpha), 1 - np.linspace(0, 1, alpha)),
        (np.linspace(0, 1, alpha), 1 - np.linspace(0, 1, alpha), np.zeros(alpha)),
        (np.linspace(0, 1, alpha), 0.5 - 0.5 * np.linspace(0, 1, alpha), 0.5 - 0.5 * np.linspace(0, 1, alpha)),
        (0.5 - 0.5 * np.linspace(0, 1, alpha), np.linspace(0, 1, alpha), 0.5 - 0.5 * np.linspace(0, 1, alpha)),
        (0.5 - 0.5 * np.linspace(0, 1, alpha), 0.5 - 0.5 * np.linspace(0, 1, alpha), np.linspace(0, 1, alpha))
    ]

    # Project the lines to 2D
    projected_lines = []
    for line in lines:
        line_3d = np.array(line)
        proj_x, proj_y = project_3d(line_3d)
        projected_lines.append((proj_x, proj_y))

    if ax is None:
        fig = plt.figure(figsize=(4, 4))
        ax = plt.axes()
        plt.rcParams['text.usetex'] = usetex
        plt.rcParams["font.style"] = 'italic'
        plt.rcParams['font.family'] = 'serif'

    if c is not None:
        ax.set_title(f"$I(X;Y) = {c:.4}$")
    ax.set_xlim([-0.1, 1.5])
    ax.set_ylim([-0.1, 1.5])
    ax.set_axis_off()

    # Plot symbols
    num_plots = len(running_model)
    condition_dim = np.shape(running_model)[0]
    if r is None:
        r = np.array([1/condition_dim]*condition_dim)

    # norm = Normalize(vmin=0, vmax=condition_dim - 1)
    norm = Normalize(vmin=0, vmax=1)

    matplotlib.colors.ColorConverter.colors['l'] = (0.3, 0.3, 0.3)

    # Plot lines
    for line_idx, (proj_x, proj_y) in enumerate(projected_lines):
        linestyle = 'k-'
        if inline and line_idx >= 3:
            linestyle = 'l--'

        ax.plot(proj_x, proj_y, linestyle, linewidth=2, zorder=0)

    print(f"{ np.shape(num_plots) = }")
    print(f"{ np.shape(condition_dim) = }")
    symbols = np.array(running_model).T
    proj_symbols_x, proj_symbols_y = project_3d(symbols)

    scatter = ax.scatter(proj_symbols_x, proj_symbols_y, c=1-r,  cmap='YlOrRd', s=70*10*r,
                marker='o', edgecolor='k', norm=norm, label='Projected Symbols')

    plt.show(block=block)


def plot_grad_dab_errors(max_n, error_vec, use_latex=False, block=False):
    plt.figure()

    plt.rcParams['text.usetex'] = use_latex
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams["font.style"] = 'italic'

    # Creating color map
    my_cmap = plt.get_cmap('YlOrRd')

    vec_len = len(error_vec)
    for i in range(vec_len):
        y = [0, 0.5, 1]

        full_x = np.array([])
        full_y = np.array([])
        full_s = np.array([])
        error = error_vec[i]
        for n in range(1, max_n + 1):
            C, r, y, iter_num, num_point, time = grad_DAB_step_error(y, n, error=error)

            s = 150 * r * (vec_len-i)

            # Fix warnings by using dtype=object
            y_array = np.array(y, dtype=object)
            x_array = np.ones(np.shape(y_array)) * n
            s_array = np.array(s)

            # Concatenate while preserving dtype
            full_x = np.concatenate((full_x, x_array), axis=0)
            full_y = np.concatenate((full_y, y_array), axis=0)
            full_s = np.concatenate((full_s, s_array), axis=0)

        label = "$\\alpha = " + str(error) + "$"
        if vec_len > 1:
            plt.scatter(full_x, full_y, s=full_s, color=my_cmap((1+i)/vec_len), label=label, alpha=1)
        else:
            def used_color(i, cmap=plt.cm.BuPu, amount=4, color_shift=2):
                return cmap((i + color_shift) / (amount + color_shift))

            amount = 5
            color_shift = 2
            cmap = plt.cm.hot
            plt.scatter(full_x, full_y, s=full_s * 2,
                        c=used_color(0, cmap, amount=amount, color_shift=color_shift), alpha=1, edgecolors='k')

    plt.xlim([0, max_n + 1])
    plt.grid()

    # 'Number of Binomial Trials - $n$'
    plt.xlabel('$n$', fontsize="18")
    # "Capacity Achieving Input Distribution - ${f}^*_X$"
    plt.ylabel("${f}^*_X$", fontsize="18")

    if vec_len > 1:
        plt.legend(fontsize="16", loc="upper left")

    plt.show(block=block)


def plot_law(scale_vec, block=True):

    max_input = 100
    plt.figure()

    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.family'] = 'serif'

    # Creating color map
    my_cmap = plt.get_cmap('YlOrRd')

    # Define the normalization for the colorbar
    vmin = scale_vec[0][0]
    vmax = scale_vec[-1][0]

    for scale_input in scale_vec:
        dim = scale_input[0]
        C_vec = scale_input[1]
        num_point_vec = scale_input[2]

        assert len(C_vec) == len(num_point_vec)
        label = '$k = '+ str(dim) +"$" # dimension
        sc = plt.plot(num_point_vec, C_vec, 's', markersize=7, mec='k', label=label, alpha=1, color=my_cmap((dim-vmin+1) / (vmax-vmin+1)))
        if num_point_vec[-1] > max_input:
            max_input = num_point_vec[-1]

    plt.plot(range(1, max_input + 1), [3/4*np.log2(a) for a in range(1, max_input + 1)], '-k', label='$ \\frac{3}{4} \log(x)$')  # , linestyle='dashdot'
    plt.plot(range(1, max_input + 1), [np.log2(a) for a in range(1, max_input + 1)], '--k', label='$\log(x)$')

    plt.xscale('log')
    plt.xlabel('$m$', fontsize="18")  # 'Support size'
    plt.ylabel('$C_{n,k}$', fontsize="18")  # Capacity
    plt.legend(fontsize="16")
    plt.grid(which="both")
    plt.xlim([1, max_input])

    plt.show(block=block)


def plot_scaling_law(scale_law_params, grad="max", threshold=1e-4, global_max_method="shgo", line_search_params=None,
                   clean_zeros=0, move_to_lines=0, block=False):

    scale_vec = []
    for i in range(len(scale_law_params)):
        param = scale_law_params[i]
        dim = param[0]
        max_n = param[1]
        print("clac multi DAB for dim = %d" % dim, " with method: " + grad)
        if line_search_params is not None:
            print("line_search: ", line_search_params[0], "constant_step: ", line_search_params[1], "min_step: ",
                  line_search_params[2], "over_shoot: ", line_search_params[3], "init: ", line_search_params[4],
                  "method: ", line_search_params[5])

        y = return_2_corners(dim=dim)
        y = [np.array(a) for a in y]
        print("input: ", y)

        iter_vec = []
        C_vec = []
        D_vec = []
        num_point_vec = []
        time_vec = []
        r_vec = []
        y_vec = []

        for n in range(1, max_n + 1):
            print(f"{n = }")
            if grad == "max" or grad == "print_kl":
                C, r, y, iter_num, D, num_point, time = multi_grad_DAB_step(y, n, init_threshold=threshold, method=grad,
                                                                            global_max_method=global_max_method,
                                                                            line_search_params=line_search_params,
                                                                            clean_zeros=clean_zeros, move_to_lines=move_to_lines)
            else:
                return -1
            iter_vec.append(iter_num)
            C_vec.append(C)
            D_vec.append(D)
            num_point_vec.append(num_point)
            time_vec.append(time)
            r_vec.append(r)
            y_vec.append(y)

        print("r_vec ", r_vec)
        print("y_vec ", y_vec)
        print("C_vec ", C_vec)
        print("num_point_vec ", num_point_vec)

        scale_vec.append([dim, C_vec, num_point_vec])
    plot_law(scale_vec, block=block)


def used_color(i, cmap=plt.cm.BuPu,  amount=4, color_shift=2):
    return cmap((i + color_shift) / (amount + color_shift))


def my_zoom_effect(ax_main, ax_zoom, lw=1.5):
    """
    Create zoom effect lines between the main plot and zoomed subplot.

    Parameters:
        ax_main (matplotlib.axes.Axes): Main plot axis.
        ax_zoom (matplotlib.axes.Axes): Zoomed subplot axis.
        lw (float): Line width of the zoom effect lines.
    """
    # Get the zoomed subplot limits
    xlim_zoom, ylim_zoom = ax_zoom.get_xlim(), ax_zoom.get_ylim()

    # Main plot data limits (rectangle region)
    xlim_main, ylim_main = ax_main.get_xlim(), ax_main.get_ylim()

    # Define corners of the zoomed region in data coordinates
    zoom_corners = [
        (xlim_zoom[0], ylim_zoom[0]),  # Bottom-left
        (xlim_zoom[0], ylim_zoom[1]),  # Top-left
        (xlim_zoom[1], ylim_zoom[0]),  # Bottom-right
        (xlim_zoom[1], ylim_zoom[1]),  # Top-right
    ]

    # Transformation pipelines
    trans_main = ax_main.transData
    trans_zoom = ax_zoom.transData
    trans_fig = ax_main.figure.transFigure.inverted()

    # Convert corners to display coordinates
    main_coords = [trans_fig.transform(trans_main.transform(corner)) for corner in zoom_corners]
    main_coords = [main_coords[1], main_coords[3]]
    zoom_coords = [trans_fig.transform(trans_zoom.transform(corner)) for corner in zoom_corners]
    zoom_coords = [zoom_coords[0], zoom_coords[2]]

    # Draw lines connecting the corners
    for (main, zoom) in zip(main_coords, zoom_coords):
        line = lines.Line2D(
            [main[0], zoom[0]], [main[1], zoom[1]],
            transform=ax_main.figure.transFigure,  # Transform applied to figure
            color="k", linestyle="--", linewidth=lw
        )
        ax_main.figure.add_artist(line)


def plot_capacity_with_zoom_20(max_n=20, use_load=False, use_latex=False, use_BA=False, plot_random=True,
                               block=True, plot_new=False):

    plt.rcParams['text.usetex'] = use_latex
    plt.rcParams['font.family'] = 'serif'

    support_size = 5
    cmap = plt.cm.hot

    amount = 8
    color_shift = 3

    n_vec = [a for a in range(1, max_n + 1)]

    # Create the figure and grid layout (1 row for zoomed-in plots, 1 row for main plot)
    fig = plt.figure(figsize=(9, 6))
    gs = GridSpec(2, 3, height_ratios=[1, 2], hspace=0.4)  # 3 small plots (row 0), 1 big plot (row 1)

    # Define zoom ranges for the small plots
    zoom_ranges = [(1, 4), (12, 14), (18, 20)]  # Example zoom-in ranges
    axes_zoom = []

    # Main Plot
    ax_main = fig.add_subplot(gs[1, :])  # Main plot spans the full width (bottom row)

    ax_main.plot(n_vec, [np.log2(support_size)] * max_n, '-.k', label=f'$\log({support_size})$')

    if use_load:
        load_information = load_support_information(use_BA=use_BA)[:max_n]
        if plot_new:
            load_information = load_support_information_multinomial(use_BA=use_BA)[:max_n]
        ax_main.plot(n_vec, load_information, c=used_color(0, cmap, amount=amount, color_shift=color_shift),
                     label=f'DeepDIVE', zorder=3)


    # result for n=13
    dab_vec = np.array([0, 0.18183863, 0.5, 0.81816137, 1])
    dab_information = fixed_support_information(dab_vec, n_vec, use_BA=use_BA)
    ax_main.plot(n_vec, dab_information, linestyle='--', c=used_color(1, cmap, amount=amount, color_shift=color_shift),
                 label=f'M-DAB$(n=13)$')

    linear_support = np.linspace(0, 1, support_size)
    squared_support = inv_square_kernel(linear_support)
    squared_information = fixed_support_information(squared_support, n_vec, use_BA=use_BA)
    ax_main.plot(n_vec, squared_information, linestyle='--',
                 c=used_color(2, cmap, amount=amount, color_shift=color_shift), label=f'Squared')

    chernoff_vec = np.array([1e-12, 0.11426210403608106, 0.5000002414182476, 0.8857384279718752, 0.999999999999])
    chernoff_information = fixed_support_information(chernoff_vec, n_vec, use_BA=use_BA)
    ax_main.plot(n_vec, chernoff_information, linestyle='--',
                 c=used_color(3, cmap, amount=amount, color_shift=color_shift), label=f'Chernoff')

    # Plot the main data
    linear_information = fixed_support_information(linear_support, n_vec, use_BA=use_BA)
    ax_main.plot(n_vec, linear_information, linestyle='--',
                 c=used_color(4, cmap, amount=amount, color_shift=color_shift), label=f'Linear')

    if plot_random:
        supports_amounts = 20
        supports = np.random.uniform(0, 1, (supports_amounts, support_size))
        informations = np.apply_along_axis(fixed_support_information, axis=1, arr=supports, n_vec=n_vec, use_BA=use_BA)
        mean_info = np.mean(informations, axis=0)
        std_info = np.std(informations, axis=0)
        ax_main.plot(n_vec, mean_info, label=f'Random', linestyle='--',
                     color=used_color(5, cmap, amount=amount, color_shift=color_shift))
        ax_main.fill_between(n_vec, mean_info - std_info, mean_info + std_info,
                             color=used_color(5, cmap, amount=amount, color_shift=color_shift), alpha=0.3)

    ax_main.set_xlim([1, max_n])
    ax_main.set_ylabel("$C_{n,k=2,d=5}$", fontsize="18")  # "Capacity"
    ax_main.set_xlabel("$n$", fontsize="18")  # Number of Multinomial Trials
    ax_main.grid(which='both')
    ax_main.legend(fontsize="12", loc="lower right")

    delta_ranges = (0.01, 0.015, 0.002)
    # Add rectangles to mark zoomed-in regions on the main plot
    rects = []
    for zoom_range, delta in zip(zoom_ranges, delta_ranges):
        # Get the data range for mean_info in the zoom_range
        y_min = np.min(chernoff_information[zoom_range[0] - 1:zoom_range[1]]) - delta
        y_max = np.max(squared_information[zoom_range[0] - 1:zoom_range[1]]) + delta

        # Create a rectangle with adjusted height
        rect = patches.Rectangle(
            (zoom_range[0], y_min),  # Bottom-left corner
            zoom_range[1] - zoom_range[0],  # Width
            y_max - y_min,  # Height
            linewidth=1.5,
            edgecolor='k',
            facecolor='none',
            linestyle='--',
        )
        ax_main.add_patch(rect)
        rects.append(rect)

    # Plot the zoomed-in regions in a row
    for i, zoom_range in enumerate(zoom_ranges):
        ax_zoom = fig.add_subplot(gs[0, i])  # Each zoomed plot occupies one column

        if use_load:
            ax_zoom.plot(n_vec, load_information, c=used_color(0, cmap, amount=amount, color_shift=color_shift),
                         label=f'DeepDIVE', zorder=3)

        ax_zoom.plot(n_vec, dab_information, linestyle='--',
                     c=used_color(1, cmap, amount=amount, color_shift=color_shift), label=f'M-DAB$(n=13)$')
        ax_zoom.plot(n_vec, squared_information, linestyle='--',
                     c=used_color(2, cmap, amount=amount, color_shift=color_shift), label=f'Squared')
        ax_zoom.plot(n_vec, chernoff_information, linestyle='--',
                     c=used_color(3, cmap, amount=amount, color_shift=color_shift), label=f'Chernoff')
        ax_zoom.plot(n_vec, linear_information, linestyle='--',
                     c=used_color(4, cmap, amount=amount, color_shift=color_shift), label=f'Linear')
        ax_zoom.plot(n_vec, [np.log2(support_size)] * max_n, '-.k', label=f'$\log({support_size})$')

        # Zoom limits
        ax_zoom.set_xlim(zoom_range)

        y_min = np.min(chernoff_information[zoom_range[0] - 1:zoom_range[1]]) - delta_ranges[i]
        y_max = np.max(squared_information[zoom_range[0] - 1:zoom_range[1]]) + delta_ranges[i]

        ax_zoom.set_ylim(y_min, y_max)
        ax_zoom.set_title(f"Zoom: $n \in [{zoom_range[0]}, {zoom_range[1]}]$", fontsize=10)
        ax_zoom.tick_params(axis='both', labelsize=8)
        ax_zoom.grid(which='both')
        axes_zoom.append(ax_zoom)

        # Apply zoom effect
        my_zoom_effect(ax_main, ax_zoom, lw=1.5)

    plt.tight_layout()
    plt.show(block=block)


def test_plot_simplex_inter_result_ax(my_x = [np.array([0, 0, 1]), np.array([0, 1/2, 1/2]), np.array([1/3, 1/3, 1/3])],
                                      n=20, ax=None, usetex=False, noise_matrix=None):

    if noise_matrix is not None:
        p_y_k, I, r = calc_pyk_noised(my_x, n, noise_matrix)
    else:
        p_y_k, I, r = calc_pyk(my_x, n, extent=False)
    print("I: ", I)
    print("r: ", r)

    def f_3d_vec(x, y, z):
        return [point_kl(n, [x[i], y[i], z[i]], p_y_k) for i in range(len(x))]

    n_x = 40
    n_y = 40
    xd = np.linspace(0, 1, n_x)
    yd = np.linspace(0, 1, n_y)
    x, y = np.meshgrid(xd, yd)

    x = np.ravel(x)
    y = np.ravel(y)
    xy = list(zip(x, y))
    triangle = list(filter(lambda a: a[0] + a[1] <= 1, xy))
    t_len = len(triangle)
    t_x, t_y = zip(*triangle)
    t_z = [1 - t_x[i]-t_y[i] for i in range(t_len)]

    if ax is None:
        fig = plt.figure()
        plt.rcParams['text.usetex'] = usetex
        plt.rcParams["font.style"] = 'italic'
        plt.rcParams['font.family'] = 'serif'
        ax = plt.axes(projection='3d', computed_zorder=False)
    ax.view_init(azim=45, elev=20)  # 50

    p3dc = ax.plot_trisurf(t_x, t_y, t_z, alpha=0.7, zorder=-2)
    mappable = map_colors(p3dc, f_3d_vec, 'YlOrRd', cmin=0, cmax=1)

    alpha = 50

    # x_0 = x_1
    zline = np.linspace(0, 1, alpha)
    yline = 0.5 - 0.5*zline
    xline = 0.5 - 0.5*zline
    ax.plot3D(xline, yline, zline, 'k', linewidth=2, alpha=1, zorder=2)

    # x_1 = x_2
    xline = np.linspace(0, 1, alpha)
    yline = 0.5 - 0.5*xline
    zline = 0.5 - 0.5*xline
    ax.plot3D(xline, yline, zline, 'k', linewidth=2, alpha=1, zorder=2)

    # x_0 = x_2
    yline = np.linspace(0, 1,  alpha)
    xline = 0.5 - 0.5*yline
    zline = 0.5 - 0.5*yline
    ax.plot3D(xline, yline, zline, 'k', linewidth=2, alpha=1, zorder=2)

    # x_2 = 0
    yline = np.linspace(0, 1,  alpha)
    xline = 1 - yline
    zline = 0*yline
    ax.plot3D(xline, yline, zline, 'k', linewidth=2, alpha=1, zorder=2)

    # x_1 = 0
    xline = np.linspace(0, 1,  alpha)
    zline = 1 - xline
    yline = 0*xline
    ax.plot3D(xline, yline, zline, 'k', linewidth=2, alpha=1, zorder=2)

    # x_0 = 0
    yline = np.linspace(0, 1,  alpha)
    zline = 1 - yline
    xline = 0*yline
    ax.plot3D(xline, yline, zline, 'k', linewidth=2, alpha=1, zorder=2)

    symbols = np.transpose(my_x)
    ax.scatter(symbols[0], symbols[1], symbols[2], c='maroon', s=70*4*r, marker='o', edgecolor='k', alpha=1,
               label='Input distribution mass point', zorder=3)  # , labal='before')

    ax.xaxis._axinfo['grid'].update(color='gray')  # X-axis grid
    ax.yaxis._axinfo['grid'].update(color='gray')  # Y-axis grid
    ax.zaxis._axinfo['grid'].update(color='gray')  # Z-axis grid

    ax.set_xlabel("$x_{1}$", fontsize="18")
    ax.set_ylabel("$x_{2}$", fontsize="18")
    ax.set_zlabel("$x_{3}$", fontsize="18")

    fig.colorbar(mappable, ax=ax, shrink=0.7)


def test_plot_simplex_inter_result(my_x=[np.array([0, 0, 1]), np.array([0, 1/2, 1/2]), np.array([1/3, 1/3, 1/3])], n=20,
                                   usetex=False, block=True):

    p_y_k, I, r = calc_pyk(my_x, n, extent=False)
    print("I: ", I)
    print("r: ", r)

    def f_3d_vec(x, y, z):
        return [point_kl(n, [x[i], y[i], z[i]], p_y_k) for i in range(len(x))]

    n_x = 40
    n_y = 40
    xd = np.linspace(0, 1, n_x)
    yd = np.linspace(0, 1, n_y)
    x, y = np.meshgrid(xd, yd)

    x = np.ravel(x)
    y = np.ravel(y)
    xy = list(zip(x, y))
    triangle = list(filter(lambda a: a[0] + a[1] <= 1, xy))
    t_len = len(triangle)
    t_x, t_y = zip(*triangle)
    t_z = [1- t_x[i]-t_y[i] for i in range(t_len)]

    fig = plt.figure()
    plt.rcParams['text.usetex'] = usetex
    plt.rcParams["font.style"] = 'italic'
    plt.rcParams['font.family'] = 'serif'
    ax = plt.axes(projection='3d', computed_zorder=False)
    ax.view_init(azim=45, elev=20)  # 50

    p3dc = ax.plot_trisurf(t_x, t_y, t_z, alpha=0.7, zorder=-2)
    mappable = map_colors(p3dc, f_3d_vec, 'YlOrRd', cmin=0, cmax=1)

    alpha = 50

    # x_0 = x_1
    zline = np.linspace(0, 1, alpha)
    yline = 0.5 - 0.5*zline
    xline = 0.5 - 0.5*zline
    ax.plot3D(xline, yline, zline, 'k', linewidth=2, alpha=1, zorder=2)

    # x_1 = x_2
    xline = np.linspace(0, 1, alpha)
    yline = 0.5 - 0.5*xline
    zline = 0.5 - 0.5*xline
    ax.plot3D(xline, yline, zline, 'k', linewidth=2, alpha=1, zorder=2)

    # x_0 = x_2
    yline = np.linspace(0, 1,  alpha)
    xline = 0.5 - 0.5*yline
    zline = 0.5 - 0.5*yline
    ax.plot3D(xline, yline, zline, 'k', linewidth=2, alpha=1, zorder=2)

    # x_2 = 0
    yline = np.linspace(0, 1,  alpha)
    xline = 1 - yline
    zline = 0*yline
    ax.plot3D(xline, yline, zline, 'k', linewidth=2, alpha=1, zorder=2)

    # x_1 = 0
    xline = np.linspace(0, 1,  alpha)
    zline = 1 - xline
    yline = 0*xline
    ax.plot3D(xline, yline, zline, 'k', linewidth=2, alpha=1, zorder=2)

    # x_0 = 0
    yline = np.linspace(0, 1,  alpha)
    zline = 1 - yline
    xline = 0*yline
    ax.plot3D(xline, yline, zline, 'k', linewidth=2, alpha=1, zorder=2)

    symbols = np.transpose(my_x)
    ax.scatter(symbols[0], symbols[1], symbols[2], c='maroon', s=70*4*r, marker='o', edgecolor='k', alpha=1, label='Input distribution mass point', zorder=3)  # , labal='before')

    ax.xaxis._axinfo['grid'].update(color='gray')  # X-axis grid
    ax.yaxis._axinfo['grid'].update(color='gray')  # Y-axis grid
    ax.zaxis._axinfo['grid'].update(color='gray')  # Z-axis grid

    ax.set_xlabel("$x_{1}$", fontsize="18")
    ax.set_ylabel("$x_{2}$", fontsize="18")
    ax.set_zlabel("$x_{3}$", fontsize="18")

    plt.show(block=block)


def plot5configurations( use_latex=False, block=True):
    x = [np.array([1, 0, 0]), np.array([0, 0, 1]), np.array([0, 1, 0]), np.array([0, 1 / 2, 1 / 2]),
         np.array([1 / 2, 0, 1 / 2])]
    test_plot_simplex_inter_result(x, n=10, block=False, usetex=use_latex)
    a = 0.4
    x = [np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([1 / 2, 1 / 2, 0]), np.array([a, 0, 1 - a]),
         np.array([0, a, 1 - a])]
    test_plot_simplex_inter_result(x, n=10, block=False, usetex=use_latex)
    plt.show(block=block)


def calc5(max_n=20, use_latex=False, block=True, plot_random=False, plot_simplex=False):
    from solvers import solve5
    eps = float_info.epsilon
    support_size = 5
    n_vec = [a for a in range(1, max_n + 1)]

    if plot_simplex:
        x = [np.array([1, 0, 0]), np.array([0, 0, 1]), np.array([0, 1, 0]), np.array([0, 1 / 2, 1 / 2]), np.array([1 / 2, 0, 1 / 2])]
        test_plot_simplex_inter_result(x, n=10, block=False, usetex=use_latex)
        a = 0.4
        x = [np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([1 / 2, 1 / 2, 0]), np.array([a, 0, 1 - a]), np.array([0, a, 1 - a])]
        test_plot_simplex_inter_result(x, n=10, block=False, usetex=use_latex)

    # Note I used also "np.array([1 / 2, 1 / 2 - eps, eps])" as Blahut-Arimoto not supported P(y_i) = 0
    corners_support = [np.array([1, 0, 0]), np.array([0, 0, 1]), np.array([0, 1, 0]), np.array([0, 1 / 2, 1 / 2]), np.array([1 / 2, 1 / 2, 0]), np.array([1 / 2, 1 / 2 - eps, eps])]
    corners_information = fixed_multinomial_support_information(corners_support, n_vec)

    middle_information = [solve5(n) for n in n_vec]

    plt.rcParams['text.usetex'] = use_latex
    plt.rcParams['font.family'] = 'serif'

    cmap = plt.cm.hot

    fig, ax_main = plt.subplots(figsize=(8, 6))
    ax_main.plot(n_vec, corners_information, c=used_color(0, cmap), label="Corners support")
    ax_main.plot(n_vec, middle_information, c=used_color(2, cmap), label="Middle support")

    if plot_random:
        supports_amounts = 20

        supports = []
        for i in range(supports_amounts):
            support = []
            for j in range(support_size):
                probs = np.random.uniform(0, 1, 2)
                probs = np.append(probs, [0, 1])
                probs.sort()
                probs = probs[1:] - probs[:-1]
                support.append(probs)
            supports.append(support)

        informations = []
        for i in range(supports_amounts):
            informations.append(fixed_multinomial_support_information(supports[i], n_vec))

        informations = np.array(informations)
        mean_info = np.mean(informations, axis=0)
        std_info = np.std(informations, axis=0)
        ax_main.plot(n_vec, mean_info, label=f'Random', color=used_color(3, cmap))
        ax_main.fill_between(n_vec, mean_info - std_info, mean_info + std_info, color=used_color(3, cmap), alpha=0.3)

    ax_main.plot(n_vec, [np.log2(support_size)]*max_n, '--k', label=f'$\log({support_size})$')
    ax_main.set_xlim([1, max_n])
    ax_main.set_ylabel("$C_{n,k=3,d=5}$", fontsize="18")  # "Capacity"
    ax_main.set_xlabel("$n$", fontsize="18")  # Number of Multinomial Trials
    ax_main.grid(which='both')
    ax_main.legend(fontsize="16", loc="lower right" )
    plt.show(block=block)


def plot_noise_confs(usetex=False, block=False):

    noise_matrix = get_alpha_matrix(n=10, alpha=0.35)

    a = 0.6
    x = [np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1]), np.array([1 / 2, 1 / 2, 0]),
         np.array([a, 0, 1 - a]), np.array([0, a, 1 - a])]

    test_plot_simplex_inter_result_ax(x, n=10, usetex=usetex, noise_matrix=noise_matrix)

    a = 0.4
    b = 0.25
    x = [np.array([1, 0, 0]), np.array([0, 1, 0]),
         np.array([1 - b, b, 0]), np.array([b, 1 - b, 0]),
         np.array([a, 0, 1 - a]), np.array([0, a, 1 - a])]
    test_plot_simplex_inter_result_ax(x, n=10, usetex=usetex, noise_matrix=noise_matrix)

    a = 0.25
    x = [np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1]),  np.array([a, 1-a, 0]), np.array([1-a, a, 0])]

    test_plot_simplex_inter_result_ax(x, n=10, usetex=usetex, noise_matrix=noise_matrix)

    plt.show(block=block)


def right_zoom_effect(ax_main, ax_zoom, lw=1.5):
    """
    Create horizontal zoom effect lines between the main plot (left) and zoomed subplot (right).
    Connects right side of zoomed region in main plot to left side of zoom plot.
    """

    xlim_zoom, ylim_zoom = ax_zoom.get_xlim(), ax_zoom.get_ylim()

    zoom_corners = [
        (xlim_zoom[0], ylim_zoom[0]),
        (xlim_zoom[0], ylim_zoom[1]),
        (xlim_zoom[1], ylim_zoom[0]),
        (xlim_zoom[1], ylim_zoom[1]),
    ]

    trans_main = ax_main.transData
    trans_zoom = ax_zoom.transData
    trans_fig = ax_main.figure.transFigure.inverted()

    main_coords = [trans_fig.transform(trans_main.transform(corner)) for corner in [zoom_corners[3], zoom_corners[2]]]
    zoom_coords = [trans_fig.transform(trans_zoom.transform(corner)) for corner in [zoom_corners[1], zoom_corners[0]]]

    for (main, zoom) in zip(main_coords, zoom_coords):
        line = lines.Line2D(
            [main[0], zoom[0]], [main[1], zoom[1]],
            transform=ax_main.figure.transFigure,
            color="k", linestyle="--", linewidth=lw
        )
        ax_main.figure.add_artist(line)


def plot_6_noised_with_fixed_zooms(num_points=11, use_latex=False, block=False):
    from solvers import solve6corners, solve6middle, solve6edge
    plt.rcParams['text.usetex'] = use_latex
    plt.rcParams['font.family'] = 'serif'

    n = 10
    alpha_vec = np.linspace(0.001, 0.4999, num_points)

    zoom_ranges = [(0.3495, 0.3505), (0.449, 0.451)]
    zoom_y_ranges = [(2.155, 2.165), (1.9, 1.93)]
    deltas = [0.005, 0.005]
    dab_val = 1.7780590425280798

    # Precompute values
    corners_vals = np.array([solve6corners(n, alpha) for alpha in alpha_vec])
    middle_vals = np.array([solve6middle(n, alpha) for alpha in alpha_vec])
    edge_vals = np.array([solve6edge(n, alpha) for alpha in alpha_vec])

    fig = plt.figure(figsize=(10, 5))
    gs = GridSpec(2, 2, width_ratios=[3, 1], height_ratios=[1, 1], hspace=0.3)

    ax_main = fig.add_subplot(gs[:, 0])  # Full left column

    amount = 5
    color_shift = 2
    cmap = plt.cm.hot

    color = used_color(0, cmap, amount=amount, color_shift=color_shift)

    # plot_noised
    num_points = 6
    alpha_vec = np.linspace(0, 0.5, num_points)

    n=10
    test_number = 6
    date = 1904
    p_file = f"results/running_p_{date}_{test_number}.pkl"
    w_file = f"results/running_w_{date}_{test_number}.pkl"

    informations = load_and_calc_noised_information_list(alpha_vec, p_file, w_file, n=n, take=10)
    informations = np.max(informations, axis=(1,2))
    informations_sparse = np.zeros(11, dtype=informations.dtype)
    informations_sparse[::2] = informations

    num_points = 11
    alpha_vec = np.linspace(0, 0.5, num_points)

    n=10
    test_number = 1
    date = 2004
    p_file = f"results/running_p_{date}_{test_number}.pkl"
    w_file = f"results/running_w_{date}_{test_number}.pkl"

    informations = load_and_calc_noised_information_list(alpha_vec, p_file, w_file, n=n , take=10)
    informations = np.max(informations, axis=(1,2))
    informations = np.maximum(informations, informations_sparse)

    ax_main.plot(alpha_vec, informations, c=color, label='DeepDIVE')

    ax_main.plot(alpha_vec, corners_vals, linestyle='--',
                 c=used_color(1, cmap, amount, color_shift), label='Corners capacity')
    ax_main.plot(alpha_vec, middle_vals, linestyle='--',
                 c=used_color(2, cmap, amount, color_shift), label='Middle capacity')
    ax_main.plot(alpha_vec, edge_vals, linestyle='--',
                 c=used_color(3, cmap, amount, color_shift), label='Edge capacity')

    ax_main.plot(alpha_vec, [dab_val] * num_points, '-.k', label='M-DAB$(n=10)$')

    ax_main.set_xlabel("$\\alpha$", fontsize=18)
    ax_main.set_ylabel("$C_{n=10,k=3,d=6}$", fontsize=18)
    ax_main.set_xlim((0, 0.5))
    ax_main.set_ylim((1.7, 2.6))
    ax_main.grid(True)
    ax_main.legend(fontsize="16",  bbox_to_anchor=(-0.00, 0.05, 0.5, 0.5))

    # Zoomed axes
    zoom_axes = [fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 1])]

    for i, (zoom_range, zoom_y_range, delta) in enumerate(zip(zoom_ranges, zoom_y_ranges, deltas)):
        x_start, x_end = zoom_range
        y_start, y_end = zoom_y_range
        alpha_min, alpha_max = alpha_vec[0], alpha_vec[-1]
        spacing = alpha_vec[1] - alpha_vec[0]

        # Convert to indices and pad with one before and after
        idx_start = max(0, int(np.floor((x_start - alpha_min) / spacing)) - 1)
        idx_end = min(len(alpha_vec) - 1, int(np.ceil((x_end - alpha_min) / spacing)) + 1)
        idx = np.arange(idx_start, idx_end + 1)

        alpha_zoom = alpha_vec[idx]
        corners_zoom = corners_vals[idx]
        middle_zoom = middle_vals[idx]
        edge_zoom = edge_vals[idx]

        # Draw rectangle in main plot
        rect = patches.Rectangle(
            (x_start, y_start), x_end - x_start, y_end - y_start,
            linewidth=1.5, edgecolor='k', facecolor='none', linestyle='--', zorder=5
        )
        ax_main.add_patch(rect)

        # Plot zoom
        ax_zoom = zoom_axes[i]
        ax_zoom.plot(alpha_vec, informations, c=color, label='DeepDIVE')
        ax_zoom.plot(alpha_zoom, corners_zoom, linestyle='--',
                     c=used_color(1, cmap, amount, color_shift), label='Corners capacity')
        ax_zoom.plot(alpha_zoom, middle_zoom, linestyle='--',
                     c=used_color(2, cmap, amount, color_shift), label='Middle capacity')
        ax_zoom.plot(alpha_zoom, edge_zoom, linestyle='--',
                     c=used_color(3, cmap, amount, color_shift), label='Edge capacity')
        ax_zoom.plot(alpha_zoom, [dab_val] * len(alpha_zoom), '--k', label='M-DAB$(n=10)$')

        ax_zoom.set_xlim(x_start, x_end)
        ax_zoom.set_ylim(y_start, y_end)
        ax_zoom.set_title(f"Zoom: $\\alpha \\in [{x_start}, {x_end}]$", fontsize=10)
        ax_zoom.tick_params(axis='both', labelsize=8)
        ax_zoom.grid(True)

        right_zoom_effect(ax_main, ax_zoom)

    plt.tight_layout()
    plt.show(block=block)


def plot6810_over_load(load_6810_list, ax_main, linestyle='-'):
    running_p_list6, running_w_list6, running_p_list8, running_w_list8, running_p_list10, running_w_list10 = load_6810_list

    information6 = calc_take_information(running_p_list6, running_w_list6, 6)
    information8 = calc_take_information(running_p_list8, running_w_list8, 8)
    information10 = calc_take_information(running_p_list10, running_w_list10, 10)

    cmap = plt.cm.hot

    def used_color(i, cmap=plt.cm.YlOrRd, amount=5, color_shift=2):  # amount=6, color_shift=2): #
        return cmap((i + color_shift) / (amount + color_shift))

    max_d = 16
    d_vec = [a for a in range(2, max_d + 1)]
    ax_main.plot(d_vec, information10, c=used_color(0, cmap), linestyle=linestyle, label=f'DeepDIVE$(n = 10)$')
    ax_main.plot(d_vec, information8, c=used_color(1, cmap), linestyle=linestyle, label=f'DeepDIVE$(n = 8)$')
    ax_main.plot(d_vec, information6, c=used_color(2, cmap), linestyle=linestyle, label=f'DeepDIVE$(n = 6)$')


def plot_over_d(block=True, plot_log=False, use_latex=True, plot_old=False):

    plt.rcParams['text.usetex'] = use_latex
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams["font.style"] = 'italic'

    fig, ax_main = plt.subplots(figsize=(8, 3))

    if plot_old:
        plot6810_over_load(load_6810(), ax_main, linestyle='--')
    plot6810_over_load(load_6810_new(), ax_main, linestyle='-')

    max_d = 16
    d_vec = [a for a in range(2, max_d + 1)]

    if plot_log:
        for support_size in d_vec:
            ax_main.plot(d_vec, [np.log2(support_size)] * len(d_vec), '--k')  # , label=f'$\log({support_size})$')

    dab_c_vec = [2.590143051510807, 2.8433049137845963, 3.0480547244282263]
    dab_color = [2,1,0]
    n_vec = [6,8,10]

    def used_color(i, cmap=plt.cm.hot, amount=5, color_shift=2):
        return cmap((i + color_shift) / (amount + color_shift))

    for dab_c, color, n in zip(dab_c_vec, dab_color, n_vec):
        ax_main.plot(d_vec, [dab_c] * len(d_vec), linestyle='--', c=used_color(color))
        # Place a label near the line (adjust the coordinates for proper placement)
        label_position = (2.2, dab_c+0.03)  # Position it around the middle of k_vec
        ax_main.text(label_position[0], label_position[1], f'M-DAB$(n={n})$', color=used_color(color), fontsize=16)

    ax_main.set_xlim([2, max_d])
    ax_main.set_ylim([1, 3.3])
    ax_main.set_ylabel("$C_{n,k=3,d}$", fontsize="18")  # "Capacity"
    ax_main.set_xlabel("$d$", fontsize="18")  # Number of Multinomial Trials
    ax_main.grid(which='both')
    ax_main.legend(fontsize="16", loc="lower right")  # bbox_to_anchor=(0.3, 0., 0.5, 0.5))
    plt.show(block=block)


def plot_over_k(block=False, plot_log=False, use_latex=False):

    cmap = plt.cm.hot

    max_k = 8
    min_k = 2
    k_vec = [a for a in range(min_k, max_k + 1)]

    plt.rcParams['text.usetex'] = use_latex
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams["font.style"] = 'italic'

    fig, ax_main = plt.subplots(figsize=(8, 3))

    n = 4
    take = 1
    date = 2004

    test_number = 11
    running_p_file = f"results/running_p_{date}_{test_number}.pkl"
    running_w_file = f"results/running_w_{date}_{test_number}.pkl"
    running_p_list = load_metrics(running_p_file)
    running_w_list = load_metrics(running_w_file)
    information32 = calc_take_information(running_p_list, running_w_list, n, take=take) # information2

    test_number = 12
    running_p_file = f"results/running_p_{date}_{test_number}.pkl"
    running_w_file = f"results/running_w_{date}_{test_number}.pkl"
    running_p_list = load_metrics(running_p_file)
    running_w_list = load_metrics(running_w_file)
    information8 = calc_take_information(running_p_list, running_w_list, n, take=take)

    test_number = 14
    running_p_file = f"results/running_p_{date}_{test_number}.pkl"
    running_w_file = f"results/running_w_{date}_{test_number}.pkl"
    running_p_list = load_metrics(running_p_file)
    running_w_list = load_metrics(running_w_file)
    information16 = calc_take_information(running_p_list, running_w_list, n, take=take)

    test_number = 15
    running_p_file = f"results/running_p_{date}_{test_number}.pkl"
    running_w_file = f"results/running_w_{date}_{test_number}.pkl"
    running_p_list = load_metrics(running_p_file)
    running_w_list = load_metrics(running_w_file)
    information64 = calc_take_information(running_p_list, running_w_list, n, take=take)

    test_number = 13
    running_p_file = f"results/running_p_{date}_{test_number}.pkl"
    running_w_file = f"results/running_w_{date}_{test_number}.pkl"
    running_p_list = load_metrics(running_p_file)
    running_w_list = load_metrics(running_w_file)
    information128 = calc_take_information(running_p_list, running_w_list, n, take=take)

    if plot_log:
        for support_size in k_vec:
            ax_main.plot(k_vec, [np.log2(support_size)] * len(k_vec), '--k')  # , label=f'$\log({support_size})$')

    dab_c_vec = [1.3723009748443582, 2.275076866894466, 2.9687148662043517, 3.539518691042205,
                 4.028023466247719, 4.4569150629945815, 4.840327962496108]
    dab_color = [0, 1, 2, 3, 4, 5, 6]

    for dab_c, color in zip(dab_c_vec[::-1], dab_color[::-1]):
        ax_main.plot(k_vec, [dab_c] * len(k_vec), linestyle='--', color='k')
        # Place a label near the line (adjust the coordinates for proper placement)
        label_position = (2.1, dab_c+0.1)  # Position it around the middle of k_vec
        ax_main.text(label_position[0], label_position[1], f'M-DAB$(k={color + 2})$', color='k', fontsize=16)

    def used_color(i, cmap=plt.cm.hot, amount=10, color_shift=2):
        return cmap((i + color_shift) / (amount + color_shift))

    ax_main.plot(k_vec, information128, c=used_color(3), linestyle='-', label=f'DeepDIVE$(d=128)$')
    ax_main.plot(k_vec, information64, c=used_color(4), linestyle='-', label=f'DeepDIVE$(d=64)$')
    ax_main.plot(k_vec, information32, c=used_color(5), linestyle='-', label=f'DeepDIVE$(d=32)$')
    ax_main.plot(k_vec, information16, c=used_color(6), linestyle='-', label=f'DeepDIVE$(d=16)$')
    ax_main.plot(k_vec, information8, c=used_color(7), linestyle='-', label=f'DeepDIVE$(d=8)$')

    ax_main.set_xlim([k_vec[0], k_vec[-1]])
    ax_main.set_ylim([1, 5.25])
    ax_main.set_ylabel("$C_{n=4,k,d}$", fontsize="18")
    ax_main.set_xlabel("$k$", fontsize="18")
    ax_main.grid(which='both')
    ax_main.legend(fontsize="16" , loc="lower right")
    plt.show(block=block)


def plot_capacity_4d_6(max_n=10, use_latex=False, use_solver=False, block=True, plot_new=False):

    eps = float_info.epsilon

    plt.rcParams['text.usetex'] = use_latex
    plt.rcParams['font.family'] = 'serif'

    support_size = 6
    n_vec = [a for a in range(1, max_n + 1)]

    amount = 5
    color_shift = 2
    cmap = plt.cm.hot
    fig, ax_main = plt.subplots(figsize=(8, 3))

    ax_main.plot(n_vec, [np.log2(support_size)] * max_n, '-.k', label=f'$\log({support_size})$')

    information = load_p_and_w_4d_6(verbose=True)
    ax_main.plot(n_vec, information, c=used_color(0, cmap, amount=amount, color_shift=color_shift), label=f'DeepDIVE')

    if plot_new:
        p_file = 'results/running_p_multi_BA_4d_6clean_0303.npy'
        w_file = 'results/running_w_multi_BA_4d_6clean_0303.npy'
        load_new = load_p_and_w_4d_6(verbose=True, p_file=p_file, w_file=w_file)
        ax_main.plot(n_vec, load_new, c=used_color(0.5, cmap, amount=amount, color_shift=color_shift), linestyle='--', label=f'new')


    uniform_support = [np.array([1, 0, 0, 0]), np.array([0, 0, 0, 1]), np.array([0, 0, 1, 0]), np.array([0, 1, 0, 0]),
                       np.array([0, 0, 1 / 2, 1 / 2]),  np.array([1 / 2, 1 / 2, 0, 0]),  np.array([1 / 2, 1 / 2-2*eps, +eps, +eps])]

    if use_solver:
        from solvers import solve
        solver_information = [solve(n=n, k=4, d=6) for n in n_vec]
        ax_main.plot(n_vec, solver_information, c=used_color(1, cmap, amount=amount, color_shift=color_shift), label=f'Solver')

    print(f"{len(uniform_support) = }")
    ax_main.plot(n_vec, fixed_multinomial_support_information(uniform_support, n_vec), c=used_color(2, cmap, amount=amount, color_shift=color_shift),
                 label="Uniform composite")

    ax_main.set_xlim([1, max_n])
    ax_main.set_ylabel("$C_{n,k=4,d=6}$", fontsize="18")  # "Capacity"
    ax_main.set_xlabel("$n$", fontsize="18")  # Number of Multinomial Trials
    ax_main.grid(which='both')
    ax_main.legend(fontsize="16", loc="lower right")  # bbox_to_anchor=(0.3, 0., 0.5, 0.5))
    plt.show(block=block)
