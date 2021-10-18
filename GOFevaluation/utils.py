from matplotlib.patches import Rectangle
from sklearn.preprocessing import KBinsDiscretizer
import numpy as np
import matplotlib as mpl


def equiprobable_histogram(data_sample, reference_sample, n_partitions,
                           order=None, plot=False, **kwargs):
    """Define equiprobable histogram based on the reference sample and
    bin the data sample according to it.

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
    bin_edges = _get_equiprobable_binning(
        reference_sample=reference_sample, n_partitions=n_partitions,
        order=order)
    n = apply_irregular_binning(data_sample=data_sample,
                                bin_edges=bin_edges,
                                order=order)
    if plot:
        plot_equiprobable_histogram(data_sample=data_sample,
                                    bin_edges=bin_edges,
                                    order=order,
                                    **kwargs)
    return n, bin_edges


def _get_equiprobable_binning(reference_sample, n_partitions, order=None):
    """Define an equiprobable binning for the reference sample. The binning
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
        enc = KBinsDiscretizer(n_bins=n_partitions, encode='ordinal',
                               strategy='quantile')
        enc.fit(np.vstack(reference_sample.T))
        bin_edges = enc.bin_edges_[0]
        bin_edges[0] = -np.inf
        bin_edges[-1] = np.inf
    else:
        if order is None:
            order = [0, 1]
        # Define data that is binned first and second based on the order argument
        first = np.vstack(reference_sample.T[order[0]])
        second = np.vstack(reference_sample.T[order[1]])

        # Get binning in first dimension:
        enc = KBinsDiscretizer(n_bins=n_partitions[order[0]], encode='ordinal',
                               strategy='quantile')
        enc.fit(first)
        bin_edges_first = enc.bin_edges_[0]
        bin_edges_first[0] = -np.inf
        bin_edges_first[-1] = np.inf

        # Get binning in second dimension (for each bin in first dimension):
        enc = KBinsDiscretizer(n_bins=n_partitions[order[1]], encode='ordinal',
                               strategy='quantile')
        bin_edges_second = []
        for low, high in zip(bin_edges_first[:-1], bin_edges_first[1:]):
            mask = (first > low) & (first <= high)
            enc.fit(np.vstack(second[mask]))
            bin_edges = enc.bin_edges_[0]
            bin_edges[0] = -np.inf
            bin_edges[-1] = np.inf
            bin_edges_second.append(bin_edges)
        bin_edges_second = np.array(bin_edges_second)
        bin_edges = [bin_edges_first, bin_edges_second]

    return bin_edges


def apply_irregular_binning(data_sample, bin_edges, order=None):
    """Apply irregular binning to data sample.

    :param data_sample: Sample of unbinned data.
    :type data_sample: array
    :param bin_edges: Array of bin edges
    :type bin_edges_first: array
    :param order: Order in which the partitioning is performed, defaults to None
        [0, 1] : first bin x then bin y for each partition in x
        [1, 0] : first bin y then bin x for each partition in y
        if None, the natural order, i.e. [0, 1] is used. For 1D just put None.
    :type order: list, optional
    :return: binned data. Number of counts in each bin.
    :rtype: array
    """
    if len(data_sample.shape) == 1:
        ns, _ = np.histogram(data_sample, bins=bin_edges)
    else:
        if order is None:
            order = [0, 1]
        first = np.vstack(data_sample.T[order[0]])
        second = np.vstack(data_sample.T[order[1]])

        ns = []
        i = 0
        for low, high in zip(bin_edges[0][:-1], bin_edges[0][1:]):
            mask = (first > low) & (first <= high)
            n, _ = np.histogram(second[mask], bins=bin_edges[1][i])
            ns.append(n)
            i += 1
    assert len(data_sample) == np.sum(ns), (f'Sum of binned data {np.sum(ns)}'
                                            + ' unequal to size of data sample'
                                            + f' {len(data_sample)}')
    return np.array(ns)


def plot_irregular_binning(ax, bin_edges, order=None, c='k', **kwargs):
    """Plot the bin edges as a grid.

    :param ax: axis to plot to
    :type ax: matplotlib axis
    :param bin_edges: Array of bin edges
    :type bin_edges: array
    :param order: Order in which the partitioning is performed, defaults to None
        [0, 1] : first bin x then bin y for each partition in x
        [1, 0] : first bin y then bin x for each partition in y
        if None, the natural order, i.e. [0, 1] is used. For 1D just put None.
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
            raise ValueError(f'order {order} is not defined.')

        i = 0
        for low, high in zip(be_first[:-1], be_first[1:]):
            if i > 0:
                plot_funcs[0](low, zorder=4, c=c, **kwargs)
            plot_funcs[1](be_second[i][1:-1], low, high, zorder=4, color=c,
                          **kwargs)
            i += 1


def plot_equiprobable_histogram(data_sample, bin_edges, order=None,
                                ax=None, nevents_expected=None, plot_xlim=None,
                                plot_ylim=None, **kwargs):
    """Plot 1d/2d histogram of data sample binned according to the passed
    irregular binning.

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
    :param plot_xlim: xlim to use for the plot. If None is passed, take min and
        max values of the data sample. Defaults to None.
    :type plot_xlim: tuple, optional
    :param plot_ylim: ylim to use for the plot. If None is passed, take min and
        max values of the data sample. Defaults to None.
    :type plot_ylim: tuple, optional
    :raises ValueError: when an unknown order is passed.
    """
    if order is None:
        order = [0, 1]
    if ax is None:
        _, ax = mpl.pyplot.subplots(1, figsize=(4, 4))
    if (plot_xlim is None) or (plot_ylim is None):
        xlim, ylim = get_plot_limits(data_sample)
    if plot_xlim is not None:
        xlim = plot_xlim
    if plot_ylim is not None:
        ylim = plot_ylim
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    ns = apply_irregular_binning(data_sample, bin_edges, order=order)

    # get colormap and norm for colorbar
    cmap_str = kwargs.get('cmap', 'RdBu_r')
    cmap = mpl.cm.get_cmap(cmap_str)
    try:
        kwargs.pop('cmap')
    except KeyError:
        pass
    if nevents_expected is None:
        norm = mpl.colors.Normalize(vmin=ns.min(), vmax=ns.max())
    else:
        n_bins = get_n_bins(bin_edges)
        midpoint = nevents_expected / n_bins
        # max deviation
        delta = max(midpoint - ns.min(), ns.max() - midpoint)
        sigma_deviation = delta / np.sqrt(midpoint)
        norm = mpl.colors.Normalize(vmin=-sigma_deviation,
                                    vmax=sigma_deviation)
        ns = (ns - midpoint) / np.sqrt(midpoint)

    if len(data_sample.shape) == 1:
        i = 0
        bin_edges[0] = xlim[0]
        bin_edges[-1] = xlim[1]
        for low, high in zip(bin_edges[:-1], bin_edges[1:]):
            rec = Rectangle((low, 0), high - low, 1,
                            facecolor=cmap(norm(ns[i])),
                            **kwargs)
            ax.add_patch(rec)
            i += 1
    else:
        be_first = bin_edges[0].copy()
        be_second = bin_edges[1].copy()

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
            raise ValueError(f'order {order} is not defined.')

        # plot rectangle for each bin
        i = 0
        edgecolor = kwargs.get('edgecolor', 'k')
        for low_f, high_f in zip(be_first[:-1], be_first[1:]):
            j = 0
            for low_s, high_s in zip(be_second[i][:-1], be_second[i][1:]):
                if order == [0, 1]:
                    rec = Rectangle((low_f, low_s),
                                    high_f - low_f,
                                    high_s - low_s,
                                    facecolor=cmap(norm(ns[i][j])),
                                    edgecolor=edgecolor,
                                    **kwargs)
                elif order == [1, 0]:
                    rec = Rectangle((low_s, low_f),
                                    high_s - low_s,
                                    high_f - low_f,
                                    facecolor=cmap(norm(ns[i][j])),
                                    edgecolor=edgecolor,
                                    **kwargs)
                ax.add_patch(rec)
                j += 1
            i += 1
    fig = mpl.pyplot.gcf()
    if nevents_expected is None:
        label = 'Counts per Bin'
    else:
        label = r'$\sigma$-deviation from expectation'
    fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax,
                 label=label)


def get_n_bins(eqpb_bin_edges):
    if isinstance(eqpb_bin_edges[0], float):
        n_bins = len(eqpb_bin_edges) - 1
    else:
        n_bins = (eqpb_bin_edges[1].shape[0]
                  * (eqpb_bin_edges[1].shape[1] - 1))
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
    assert ~np.isnan(sample).any(), 'Sample contains NaN entries!'
    assert ~np.isinf(sample).any(), 'Sample contains inf values!'
