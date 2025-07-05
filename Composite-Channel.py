from visualize import plot_simplex_maximizer_example, compare_capacity_4d, plot_grad_dab_errors, plot_scaling_law, \
    plot_capacity_with_zoom_20, plot5configurations, plot_noise_confs, plot_6_noised_with_fixed_zooms, \
    plot_over_d, plot_over_k, plot_capacity_4d_6
from solvers import plot_local_minimas


def generate_all_figures():
    # Figure 1: The first iteration of the M-DAB algorithm for C_{n=7,k=3}.
    plot_simplex_maximizer_example()  # distance measure effect

    # Figure 3: Mutual information achieved by different constellations for C_{n=6,k=3}.
    plot_local_minimas()  # configuration for d=10

    # Figure 4: M-DAB results for k = 2 and symmetric noise parameter \alpha.
    plot_grad_dab_errors(max_n=50, error_vec=[0, 0.05, 0.1, 0.15, 0.2], use_latex=True)

    # Figure 5: M-DAB results for multinomial channel.
    # (a) Capacity achieved for C_{n,k=4}
    compare_capacity_4d()  # composite vs uniform
    # (b) The capacity as a function of the number of mass points in the minimal support size.
    line_search_params = [1, 0, 5e-2, 1, 1e-1, "shgo"]
    params = [[2, 35], [3, 20], [4, 10], [5, 10]]
    plot_scaling_law(params, threshold=1e-4, global_max_method="hand", line_search_params=line_search_params,
                    clean_zeros=1e-3, move_to_lines=1e-3, block=True)

    # Figure 6: DeepDIVE’s results compared to previous methods.

    # (a) Geometric and probabilistic shaping
    plot_capacity_with_zoom_20(20, use_load=True, use_latex=True, use_BA=True, plot_random=True,
                               block=False, plot_new=True)
    # (b) Only geometric shaping
    plot_capacity_with_zoom_20(20, use_load=True, use_latex=True, use_BA=False, plot_random=True, block=False)

    # Figure 7: Five-symbols constellations on two-dimensional simplex.
    # (a) Corners configuration (b) Middle configuration
    plot5configurations(use_latex=True, block=False)

    # Figure 8: Six-symbols constellations on two-dimensional simplex.
    # (a) Corners configuration (b) Middle configuration (c) Edge configuration
    plot_noise_confs(usetex=True, block=False)

    # Figure 9: Mutual information achieved by the corners, middle, and edge configurations compared to DeepDIVE
    plot_6_noised_with_fixed_zooms(num_points=11, use_latex=True, block=True)

    # Figure 10: DeepDIVE’s performance under varying parameters
    # (a) Varying input-support constraint d
    plot_over_d(block=True, use_latex=True)
    # (b) Varying channel dimension k
    plot_over_k(block=True, use_latex=True)

    # Figure 11: Comparison of mutual information for six-letter composite alphabets.
    plot_capacity_4d_6(block=True, use_latex=True)


if __name__ == "__main__":
    generate_all_figures()
