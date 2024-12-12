import warnings
from matplotlib.patches import Rectangle
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from copy import deepcopy


def equiprobable_histogram(
    data_sample,
    reference_sample,
    n_partitions,
    order=None,
    plot=False,
    reference_sample_weights=None,
    data_sample_weights=None,
    **kwargs,
):
    """Define equiprobable histogram based on the reference sample and bin the data
    sample according to it.

    :param data_sample: Sample of unbinned data.
    :type data_sample: array
    :param reference_sample: sample of unbinned reference
    :type reference_sample: array_like, n-Dimensional
    :param n_partitions: Number of partitions in each dimension
    :type n_partitions: list of int
    :param order: Order in which the partitioning is performed, defaults to None
        [0, 1] : first bin x then bin y for each partition in x
        [1, 0] : first bin y then bin x for each partition in y
        if None, the natural order, i.e. [0, 1] is used. For 1D just put None.
    :type order: list, optional
    :param plot: if True, histogram of data sample is plotted, defaults to False
    :type plot: bool, optional
    :param reference_sample_weights: weights of reference_sample
    :type reference_sample_weights: array_like, 1-Dimensional
    :param data_sample_weights: weights of data_sample
    :type data_sample_weights: array_like, 1-Dimensional
    :return: n, bin_edges
        n: number of counts of data sample in each bin
        bin_edges: For order [0, 1]([1, 0])
        these are the bin edges in x(y) and y(x) respectively. bin_edges[1]
        is a list of bin edges corresponding to the partitions defined in
        bin_edges[0].
    .. note::
        Reference: F. James, 2008: "Statistical Methods in Experimental
                    Physics", Ch. 11.2.3

    """
    check_dimensionality_for_eqpb(data_sample, reference_sample, n_partitions, order)
    bin_edges = get_equiprobable_binning(
        reference_sample=reference_sample,
        n_partitions=n_partitions,
        order=order,
        reference_sample_weights=reference_sample_weights,
    )
    n = apply_irregular_binning(
        data_sample=data_sample,
        bin_edges=bin_edges,
        order=order,
        data_sample_weights=data_sample_weights,
    )
    if plot:
        plot_equiprobable_histogram(
            data_sample=data_sample,
            reference_sample=reference_sample,
            bin_edges=bin_edges,
            order=order,
            data_sample_weights=data_sample_weights,
            reference_sample_weights=reference_sample_weights,
            **kwargs,
        )
    return n, bin_edges


def _get_finite_bin_edges(bin_edges, data_sample, order):
    """Replaces infinite values in bin_edges with finite values determined such that
    the bins encompass all the counts in data_sample.

    Necessary for plotting
    and for determining bin area.
    :param bin_edges: list of bin_edges,
    probably form get_equiprobable_binning
    :type bin_edges: array
    :param data_sample: Sample of unbinned data.
    :type data_sample: array
    :param order: Order in which the partitioning is performed,
    defaults to None
        [0, 1] : first bin x then bin y for each partition in x
        [1, 0] : first bin y then bin x for each partition in y
        if None, the natural order, i.e. [0, 1] is used. For 1D just put None.
    :type order: list, optional
    :return: Returns bin_edges.
        1D: list of bin edges
        2D: For order [0, 1]([1, 0]) these are the bin edges in x(y) and y(x)
        respectively. bin_edges[1] is a list of bin edges corresponding to the
        partitions defined in bin_edges[0].
    :rtype: list of arrays

    """
    xlim, ylim = get_plot_limits(data_sample)
    be = []
    if len(data_sample.shape) == 1:
        be = [deepcopy(bin_edges), None]
        be[0][0] = xlim[0]
        be[0][-1] = xlim[1]
    else:
        be_first = deepcopy(bin_edges[0])
        be_second = deepcopy(bin_edges[1])

        if order == [0, 1]:
            be_first[be_first == -np.inf] = xlim[0]
            be_first[be_first == np.inf] = xlim[1]
            be_second[be_second == -np.inf] = ylim[0]
            be_second[be_second == np.inf] = ylim[1]
        elif order == [1, 0]:
            be_first[be_first == -np.inf] = ylim[0]
            be_first[be_first == np.inf] = ylim[1]
            be_second[be_second == -np.inf] = xlim[0]
            be_second[be_second == np.inf] = xlim[1]
        else:
            raise ValueError(f"order {order} is not defined.")
        be = [be_first, be_second]

    return be


def _get_count_density(ns, be_first, be_second, data_sample):
    """Measures the area of each bin and scales the counts in that bin by the inverse
    of that area.

    :param be_first: list of bin_edges in the first dimension,
    :type be_first: array
    :param be_first: list of bin_edges in the first dimension,
    :type be_first: array
    :param data_sample: Sample of unbinned data.
    :type data_sample: array
    :param order: Order in which the partitioning is performed,
        defaults to None
        [0, 1] : first bin x then bin y for each partition in x
        [1, 0] : first bin y then bin x for each partition in y
        if None, the natural order, i.e. [0, 1] is used. For 1D just put None.
    :type order: list, optional
    :return: Returns bin_edges.
        1D: list of bin edges
        2D: For order [0, 1]([1, 0]) these are the bin edges in x(y) and y(x)
        respectively. bin_edges[1] is a list of bin edges corresponding to the
        partitions defined in bin_edges[0].
    :rtype: list of arrays

    """
    if len(data_sample.shape) == 1:
        i = 0
        for low, high in zip(be_first[:-1], be_first[1:]):
            ns[i] = ns[i] / (high - low)
            i += 1
    else:
        i = 0
        for low_f, high_f in zip(be_first[:-1], be_first[1:]):
            j = 0
            for low_s, high_s in zip(be_second[i][:-1], be_second[i][1:]):
                ns[i][j] = ns[i][j] / ((high_f - low_f) * (high_s - low_s))
                j += 1
            i += 1

    return ns


def _equi(n_bins, reference_sample):
    """Perform a 1D equiprobable binning for reference_sample.

    :param n_bins: number of partitions in this dimension
    :type n_bins: int
    :param reference_sample: sample of unbinned reference
    :type reference_sample: array_like, 1-Dimensional
    :return: Returns bin_edges.
    :rtype: array_like, 1-Dimensional

    """
    bin_edges = np.quantile(reference_sample, np.linspace(0, 1, n_bins + 1)[1:-1])
    bin_edges = np.hstack([-np.inf, bin_edges, np.inf])
    return bin_edges


def _weighted_equi(n_bins, reference_sample, reference_sample_weights):
    """Perform a 1D equiprobable binning for reference_sample with weights.

    :param n_bins: number of partitions in this dimension
    :type n_bins: int
    :param reference_sample: sample of unbinned reference
    :type reference_sample: array_like, 1-Dimensional
    :param reference_sample_weights: weights of reference_sample
    :type reference_sample_weights: array_like, 1-Dimensional
    :return: Returns bin_edges.
    :rtype: array_like, 1-Dimensional

    """
    argsort = reference_sample.argsort()
    reference_sample_weights = reference_sample_weights[argsort]
    reference_sample = reference_sample[argsort]
    cumsum = np.cumsum(reference_sample_weights)
    cumsum -= cumsum[0]
    bin_edges = np.interp(
        np.linspace(0, 1, n_bins + 1)[1:-1], cumsum / cumsum[-1], reference_sample
    )
    bin_edges = np.hstack([-np.inf, bin_edges, np.inf])
    return bin_edges


def get_equiprobable_binning(
    reference_sample, n_partitions, order=None, reference_sample_weights=None
):
    """Define an equiprobable binning for the reference sample.

    The binning
    is defined such that the number of counts in each bin are (almost) equal.
    Bins are defined based on the ECDF of the reference sample.
    The number of partitions in x and y direction as well as the order of
    partitioning influence the result.
    :param reference_sample: sample of unbinned reference
    :type reference_sample: array_like, n-Dimensional
    :param n_partitions: Number of partitions in each dimension
    :type n_partitions: list of int
    :param order: Order in which the partitioning is performed, defaults to None
        [0, 1] : first bin x then bin y for each partition in x
        [1, 0] : first bin y then bin x for each partition in y
        if None, the natural order, i.e. [0, 1] is used. For 1D just put None.
    :type order: list, optional
    :param reference_sample_weights: weights of reference_sample
    :type reference_sample_weights: array_like, 1-Dimensional
    :return: Returns bin_edges.
        1D: list of bin edges
        2D: For order [0, 1]([1, 0]) these are the bin edges in x(y) and y(x)
        respectively. bin_edges[1] is a list of bin edges corresponding to the
        partitions defined in bin_edges[0].
    :rtype: list of arrays
    :raises ValueError: when an unknown order is passed.
    .. note::
        Reference: F. James, 2008: "Statistical Methods in Experimental
                    Physics", Ch. 11.2.3

    """
    if len(reference_sample.shape) == 1:
        dim = 1
    elif len(reference_sample.shape) == 2:
        dim = 2
    else:
        raise TypeError(f"reference_sample has unsupported shape {reference_sample.shape}.")
    check_for_ties(reference_sample, dim=dim)

    if reference_sample_weights is None:
        weights_flag = 0
    else:
        _check_weight_sanity(reference_sample, reference_sample_weights)
        weights_flag = 1
    if dim == 1:
        if weights_flag:
            bin_edges = _weighted_equi(n_partitions, reference_sample, reference_sample_weights)
        else:
            bin_edges = _equi(n_partitions, reference_sample)
    elif dim == 2:
        if order is None:
            order = [0, 1]
        first = reference_sample.T[order[0]]
        second = reference_sample.T[order[1]]
        if weights_flag:
            bin_edges_first = _weighted_equi(
                n_partitions[order[0]], first, reference_sample_weights
            )
        else:
            bin_edges_first = _equi(n_partitions[order[0]], first)

        # Get binning in second dimension (for each bin in first dimension):
        bin_edges_second = []
        for low, high in zip(bin_edges_first[:-1], bin_edges_first[1:]):
            mask = (first >= low) & (first < high)
            if weights_flag:
                bin_edges = _weighted_equi(
                    n_partitions[order[1]], second[mask], reference_sample_weights[mask]
                )
            else:
                bin_edges = _equi(n_partitions[order[1]], second[mask])
            bin_edges_second.append(bin_edges)
        bin_edges_second = np.array(bin_edges_second)
        bin_edges = [bin_edges_first, bin_edges_second]
    return bin_edges


def _check_weight_sanity(reference_sample, reference_sample_weights):
    """Check if the weights are larger than 0, and if reference has the same shape to
    the weights."""
    mesg = "data and their weights should be in the same length"
    assert len(reference_sample) == len(reference_sample_weights), mesg

    mesg = "weights should be 1D array"
    assert len(reference_sample_weights.shape) == 1, mesg

    mesg = "all weights should be non-negative"
    assert np.all(reference_sample_weights >= 0), mesg


def apply_irregular_binning(data_sample, bin_edges, order=None, data_sample_weights=None):
    """Apply irregular binning to data sample.

    :param data_sample: Sample of unbinned data.
    :type data_sample: array
    :param bin_edges: Array of bin edges
    :type bin_edges_first: array
    :param order: Order in which the partitioning is performed, defaults to None [0, 1]
        : first bin x then bin y for each partition in x [1, 0] : first bin y then bin
        x for each partition in y if None, the natural order, i.e. [0, 1] is used. For
        1D just put None.
    :type order: list, optional
    :param data_sample_weights: weights of data_sample
    :type data_sample_weights: array_like, 1-Dimensional
    :return: binned data. Number of counts in each bin.
    :rtype: array

    """
    if data_sample_weights is None:
        weights_flag = 0
    else:
        _check_weight_sanity(data_sample, data_sample_weights)
        weights_flag = 1
    if len(data_sample.shape) == 1:
        ns, _ = np.histogram(data_sample, bins=bin_edges, weights=data_sample_weights)
    else:
        if order is None:
            order = [0, 1]
        first = np.vstack(data_sample.T[order[0]])
        second = np.vstack(data_sample.T[order[1]])

        ns = []
        i = 0
        for low, high in zip(bin_edges[0][:-1], bin_edges[0][1:]):
            mask = (first >= low) & (first < high)
            if weights_flag:
                n, _ = np.histogram(
                    second[mask], bins=bin_edges[1][i], weights=data_sample_weights[mask.flatten()]
                )
            else:
                n, _ = np.histogram(second[mask], bins=bin_edges[1][i])
            ns.append(n)
            i += 1
    if weights_flag:
        mesg = (
            f"Sum of binned data {np.sum(ns)}"
            + " unequal to sum of data weights"
            + f" {np.sum(data_sample_weights)}"
        )
        assert np.isclose(np.sum(data_sample_weights), np.sum(ns), rtol=1e-3), mesg
    else:
        mesg = (
            f"Sum of binned data {np.sum(ns)}"
            + " unequal to size of data sample"
            + f" {len(data_sample)}"
        )
        assert len(data_sample) == np.sum(ns), mesg
    return np.array(ns, dtype=float)


def plot_irregular_binning(ax, bin_edges, order=None, c="k", **kwargs):
    """Plot the bin edges as a grid.

    :param ax: axis to plot to
    :type ax: matplotlib axis
    :param bin_edges: Array of bin edges
    :type bin_edges: array
    :param order: Order in which the partitioning is performed, defaults to None [0, 1]
        : first bin x then bin y for each partition in x [1, 0] : first bin y then bin
        x for each partition in y if None, the natural order, i.e. [0, 1] is used. For
        1D just put None.
    :type order: list, optional
    :param c: color of the grid, defaults to 'k'
    :type c: str, optional
    :param kwargs: kwargs are passed to the plot functions
    :raises ValueError: when an unknown order is passed.

    """
    if order is None:
        order = [0, 1]
    if bin_edges[0].shape == ():
        for line in bin_edges:
            ax.axvline(line, c=c, zorder=4, **kwargs)
    else:
        ylim = ax.get_ylim()
        xlim = ax.get_xlim()
        be_first = bin_edges[0].copy()
        be_second = bin_edges[1].copy()

        if order == [0, 1]:
            plot_funcs = [ax.axvline, ax.hlines]
            be_first[be_first == -np.inf] = xlim[0]
            be_first[be_first == np.inf] = xlim[1]
            be_second[be_second == -np.inf] = ylim[0]
            be_second[be_second == np.inf] = ylim[1]
        elif order == [1, 0]:
            plot_funcs = [ax.axhline, ax.vlines]
            be_first[be_first == -np.inf] = ylim[0]
            be_first[be_first == np.inf] = ylim[1]
            be_second[be_second == -np.inf] = xlim[0]
            be_second[be_second == np.inf] = xlim[1]
        else:
            raise ValueError(f"order {order} is not defined.")

        i = 0
        for low, high in zip(be_first[:-1], be_first[1:]):
            if i > 0:
                plot_funcs[0](low, zorder=4, c=c, **kwargs)
            plot_funcs[1](be_second[i][1:-1], low, high, zorder=4, color=c, **kwargs)
            i += 1


def plot_equiprobable_histogram(
    data_sample,
    bin_edges,
    order=None,
    reference_sample=None,
    ax=None,
    nevents_expected=None,
    data_sample_weights=None,
    reference_sample_weights=None,
    plot_xlim=None,
    plot_ylim=None,
    plot_mode="sigma_deviation",
    draw_colorbar=True,
    **kwargs,
):
    """Plot 1d/2d histogram of data sample binned according to the passed irregular
    binning.

    :param data_sample: Sample of unbinned data.
    :type data_sample: array
    :param bin_edges: Array of bin edges
    :type bin_edges: array
    :param order: Order in which the partitioning is performed
        [0, 1] : first bin x then bin y for each partition in x
        [1, 0] : first bin y then bin x for each partition in y
        if None, the natural order, i.e. [0, 1] is used. For 1D just put None.
    :type order: list, optional
    :param ax: axis to plot to, if None: make new axis. Defaults to None.
    :type ax: matplotlib axis, optional
    :param nevents_expected: total number of expected events used for centering
        the colormap around the expectation value per bin and giving the
        z-axis in units of sigma-deviation from expectation. If None is passed,
        cmap scale ranges from min to max. Defaults to None.
    :type nevents_expected: float, optional
    :param plot_mode: sets the plotting schedule. Defaults to sigma_deviation
        which shows the deviation of the counts in each bin from expected.
        Can be set to 'num_counts' to plot the total number of counts in each
        bin or 'count_density' to show the counts scaled by the inverse of the
        area of the bin, throws error if set to other value
    :type plot_mode: string, optional
    :param draw_colorbar: whether draw the colorbar
    :type draw_colorbar: bool, optional
    :param plot_xlim: xlim to use for the plot. If None is passed, take min and
        max values of the data sample. Defaults to None.
    :type plot_xlim: tuple, optional
    :param plot_ylim: ylim to use for the plot. If None is passed, take min and
        max values of the data sample. Defaults to None.
    :type plot_ylim: tuple, optional
    :raises ValueError: when an unknown order is passed.

    """
    # Setup plot
    if order is None:
        order = [0, 1]
    if ax is None:
        _, ax = plt.subplots(1, figsize=(4, 4))
    if (plot_xlim is None) or (plot_ylim is None):
        xlim, ylim = get_plot_limits(data_sample)
    if plot_mode == "count_density":
        if (plot_xlim is not None) or (plot_ylim is not None):
            raise RuntimeError("Manually set x or y limit in" "count_density mode is misleading")
    if plot_xlim is not None:
        xlim = plot_xlim
    if plot_ylim is not None:
        ylim = plot_ylim

    # bin data and reference sample
    ns = apply_irregular_binning(
        data_sample=data_sample,
        bin_edges=bin_edges,
        order=order,
        data_sample_weights=data_sample_weights,
    )
    if reference_sample is not None:
        pdf = apply_irregular_binning(
            data_sample=reference_sample,
            bin_edges=bin_edges,
            order=order,
            data_sample_weights=reference_sample_weights,
        )
        pdf = pdf / np.sum(pdf)
    else:
        pdf = None

    be = _get_finite_bin_edges(bin_edges, data_sample, order)
    be_first = be[0]
    be_second = be[1]

    alpha = kwargs.pop("alpha", 1)

    # format according to plot_mode
    if plot_mode == "sigma_deviation":
        cmap_str = kwargs.pop("cmap", "RdBu_r")
        cmap = _get_cmap(cmap_str, alpha=alpha)
        if nevents_expected is None:
            raise ValueError("nevents_expected cannot " "be None while plot_mode='sigma_deviation'")
        if reference_sample is None:
            raise ValueError("reference_sample cannot " "be None while plot_mode='sigma_deviation'")
        ns_expected = nevents_expected * pdf
        ns = (ns - ns_expected) / np.sqrt(ns_expected)
        max_deviation = max(np.abs(ns.ravel()))
        vmin = -max_deviation
        vmax = max_deviation
        if abs(kwargs.get("vmin", vmin)) != abs(kwargs.get("vmax", vmax)):
            warnings.warn("You are specifying different `vmin` and `vmax`!", stacklevel=2)
        if np.allclose(ns_expected.ravel(), ns_expected.ravel()[0], rtol=1e-4, atol=1e-2):
            midpoint = ns_expected.ravel()[0]
            label = r"$\sigma$-deviation from $\mu_\mathrm{{bin}}$ =" + f"{midpoint:.1f} counts"
        else:
            warnings.warn(
                "The expected counts in the bins are not equal, "
                f"ranging from {np.min(ns_expected)} to "
                f"{np.max(ns_expected)}.",
                stacklevel=2,
            )
            label = r"$\sigma$-deviation from $\mu_\mathrm{{bin}}$"
    elif plot_mode == "count_density":
        label = r"Counts per area in each bin"
        ns = _get_count_density(ns, be_first, be_second, data_sample)
        cmap_str = kwargs.pop("cmap", "viridis")
        cmap = _get_cmap(cmap_str, alpha=alpha)
        vmin = np.min(ns)
        vmax = np.max(ns)
    elif plot_mode == "num_counts":
        label = r"Number of counts in each bin"
        cmap_str = kwargs.pop("cmap", "viridis")
        cmap = _get_cmap(cmap_str, alpha=alpha)
        vmin = np.min(ns)
        vmax = np.max(ns)
    else:
        raise ValueError(f"plot_mode {plot_mode} is not defined.")

    # draw bins
    norm = mpl.colors.Normalize(
        vmin=kwargs.pop("vmin", vmin), vmax=kwargs.pop("vmax", vmax), clip=False
    )
    edgecolor = kwargs.pop("edgecolor", "k")
    if len(data_sample.shape) == 1:
        i = 0
        be_first[0] = xlim[0]
        be_first[-1] = xlim[1]
        for low, high in zip(be_first[:-1], be_first[1:]):
            ax.axvspan(
                low,
                high,
                facecolor=cmap(norm(ns[i])),
                edgecolor=edgecolor,
                **kwargs,
            )
            i += 1
    else:
        # plot rectangle for each bin
        i = 0
        if order == [0, 1]:
            be_first[0] = xlim[0]
            be_first[-1] = xlim[1]
        elif order == [1, 0]:
            be_first[0] = ylim[0]
            be_first[-1] = ylim[1]
        for low_f, high_f in zip(be_first[:-1], be_first[1:]):
            j = 0
            if order == [0, 1]:
                be_second[i][0] = ylim[0]
                be_second[i][-1] = ylim[1]
            elif order == [1, 0]:
                be_second[i][0] = xlim[0]
                be_second[i][-1] = xlim[1]
            for low_s, high_s in zip(be_second[i][:-1], be_second[i][1:]):
                if order == [0, 1]:
                    rec = Rectangle(
                        (low_f, low_s),
                        high_f - low_f,
                        high_s - low_s,
                        facecolor=cmap(norm(ns[i][j])),
                        edgecolor=edgecolor,
                        **kwargs,
                    )
                elif order == [1, 0]:
                    rec = Rectangle(
                        (low_s, low_f),
                        high_s - low_s,
                        high_f - low_f,
                        facecolor=cmap(norm(ns[i][j])),
                        edgecolor=edgecolor,
                        **kwargs,
                    )
                ax.add_patch(rec)
                j += 1
            i += 1

    # Cosmetics
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    if draw_colorbar:
        fig = plt.gcf()

        extend = "neither"
        if norm.vmin > np.min(ns) and norm.vmax < np.max(ns):
            extend = "both"
        elif norm.vmin > np.min(ns):
            extend = "min"
        elif norm.vmax < np.max(ns):
            extend = "max"

        fig.colorbar(
            mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
            ax=ax,
            label=label,
            extend=extend,
        )
    return


def get_n_bins(eqpb_bin_edges):
    if isinstance(eqpb_bin_edges[0], float):
        n_bins = len(eqpb_bin_edges) - 1
    else:
        n_bins = eqpb_bin_edges[1].shape[0] * (eqpb_bin_edges[1].shape[1] - 1)
    return n_bins


def get_plot_limits(data_sample):
    if len(data_sample.shape) == 1:
        xlim = (min(data_sample), max(data_sample))
        ylim = None
    else:
        xlim = (min(data_sample.T[0]), max(data_sample.T[0]))
        ylim = (min(data_sample.T[1]), max(data_sample.T[1]))

    return xlim, ylim


def check_sample_sanity(sample):
    assert ~np.isnan(sample).any(), "Sample contains NaN entries!"
    assert ~np.isinf(sample).any(), "Sample contains inf values!"


def check_for_ties(sample, dim):
    if dim == 1:
        any_ties = len(np.unique(sample)) != len(sample)
    elif dim == 2:
        any_ties = len(np.unique(sample.T[0])) != len(sample.T[0])
        any_ties |= len(np.unique(sample.T[1])) != len(sample.T[1])
    else:
        raise ValueError(f"dim {dim} is not defined.")
    if any_ties:
        warnings.warn(
            "reference_sample contains ties, this might "
            "cause problems in the equiprobable binning.",
            stacklevel=2,
        )


def check_dimensionality_for_eqpb(data_sample, reference_sample, n_partitions, order):
    if len(reference_sample.shape) == 1:
        assert len(data_sample.shape) == 1, (
            "Shape of data_sample is" " incompatible with shape of reference_sample"
        )
        assert isinstance(n_partitions, int), "n_partitions must be an" " integer for 1-dim. data."
        assert order is None, (
            "providing a not-None value for order is" " ambiguous for 1-dim. data."
        )
    elif len(reference_sample.shape) == 2:
        assert len(data_sample.shape) == 2, (
            "Shape of data_sample is" " incompatible with shape of reference_sample."
        )
        # Check dimensionality is two
        assert data_sample.shape[1] == reference_sample.shape[1] == len(n_partitions), (
            "Shape of data_sample is incompatible with shape of"
            " reference_sample and/or dimensionality of n_partitions."
        )
        if data_sample.shape[1] > 2:
            raise NotImplementedError(
                "Equiprobable binning is not (yet) "
                f"implemented for {data_sample.shape[1]}"
                "-dimensional data."
            )
    else:
        raise TypeError("reference_sample has unsupported shape " f"{reference_sample.shape}.")


def _get_cmap(cmap_str, alpha=1):
    _cmap = mpl.colormaps[cmap_str]
    cmap = _cmap(np.arange(_cmap.N))
    cmap[:, -1] = alpha
    return mpl.colors.LinearSegmentedColormap.from_list("dummy", cmap)
