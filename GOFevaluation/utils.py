import warnings
from matplotlib.patches import Rectangle
from sklearn.preprocessing import KBinsDiscretizer
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


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


def _get_finite_bin_edges(bin_edges, data_sample, order):
    """Replaces infinite values in bin_edges with finite
    values determined such that the bins encompass all
    the counts in data_sample. Necessary for plotting
    and for determining bin area.
    :param bin_edges: list of bin_edges,
    probably form _get_equiprobable_binning
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
        bin_edges[0] = xlim[0]
        bin_edges[-1] = xlim[1]
        be = [bin_edges.copy(), None]
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
        be = [be_first, be_second]

    return be


def _get_count_density(ns, be_first, be_second, data_sample):
    """Measures the area of each bin and scales the counts in
    that bin by the inverse of that area.
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
    return np.array(ns, dtype=float)


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
                                ax=None, nevents_expected=None,
                                plot_xlim=None, plot_ylim=None,
                                plot_mode='sigma_deviation',
                                draw_colorbar=True, **kwargs):
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
    if order is None:
        order = [0, 1]
    if ax is None:
        _, ax = plt.subplots(1, figsize=(4, 4))
    if (plot_xlim is None) or (plot_ylim is None):
        xlim, ylim = get_plot_limits(data_sample)
    if plot_mode == 'count_density':
        if (plot_xlim is not None) or (plot_ylim is not None):
            raise RuntimeError('Manually set x or y limit in'
                               'count_density mode is misleading')
    if plot_xlim is not None:
        xlim = plot_xlim
    if plot_ylim is not None:
        ylim = plot_ylim

    ns = apply_irregular_binning(data_sample, bin_edges, order=order)

    be = _get_finite_bin_edges(bin_edges, data_sample, order)
    be_first = be[0]
    be_second = be[1]

    if plot_mode == 'sigma_deviation':
        cmap_str = kwargs.pop('cmap', 'RdBu_r')
        cmap = mpl.cm.get_cmap(cmap_str)
        if nevents_expected is None:
            raise ValueError('nevents_expected cannot ' +
                             'be None while plot_mode=\'sigma_deviation\'')
        n_bins = get_n_bins(bin_edges)
        midpoint = nevents_expected / n_bins
        delta = max(midpoint - ns.min(), ns.max() - midpoint)
        sigma_deviation = delta / np.sqrt(midpoint)
        ns = (ns - midpoint) / np.sqrt(midpoint)
        vmin = -sigma_deviation
        vmax = sigma_deviation
        if abs(kwargs.get('vmin', vmin)) != abs(kwargs.get('vmax', vmax)):
            warnings.warn('You are specifying different `vmin` and `vmax`!',
                          stacklevel=2)
        label = (r'$\sigma$-deviation from $\mu_\mathrm{{bin}}$ ='
                 + f'{midpoint:.1f} counts')
    elif plot_mode == 'count_density':
        label = r'Counts per area in each bin'
        ns = _get_count_density(ns, be_first, be_second, data_sample)
        cmap_str = kwargs.pop('cmap', 'viridis')
        cmap = mpl.cm.get_cmap(cmap_str)
        vmin = np.min(ns)
        vmax = np.max(ns)
    elif plot_mode == 'num_counts':
        label = r'Number of counts in eace bin'
        cmap_str = kwargs.pop('cmap', 'viridis')
        cmap = mpl.cm.get_cmap(cmap_str)
        vmin = np.min(ns)
        vmax = np.max(ns)
    else:
        raise ValueError(f'plot_mode {plot_mode} is not defined.')

    norm = mpl.colors.Normalize(vmin=kwargs.pop('vmin', vmin),
                                vmax=kwargs.pop('vmax', vmax),
                                clip=False)

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
        # plot rectangle for each bin
        i = 0
        if order == [0, 1]:
            be_first[0] = xlim[0]
            be_first[-1] = xlim[1]
        elif order == [1, 0]:
            be_first[0] = ylim[0]
            be_first[-1] = ylim[1]
        edgecolor = kwargs.pop('edgecolor', 'k')
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

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    if draw_colorbar:
        fig = plt.gcf()

        extend = 'neither'
        if norm.vmin > np.min(ns) and norm.vmax < np.max(ns):
            extend = 'both'
        elif norm.vmin > np.min(ns):
            extend = 'min'
        elif norm.vmax < np.max(ns):
            extend = 'max'

        fig.colorbar(
            mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
            ax=ax,
            label=label,
            extend=extend
        )
    return


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
