from matplotlib.patches import Rectangle
from sklearn.preprocessing import KBinsDiscretizer
import numpy as np
import matplotlib as mpl


def get_equiprobable_binning(reference_sample, n_part_x, n_part_y=None,
                             order='xy'):
    """Define an equiprobable binning for the reference sample. The binning
    is defined such that the number of counts in each bin are (almost) equal.
    Bins are defined based on the ECDF of the reference sample.
    The number of partitions in x and y direction as well as the order of
    partitioning influence the result.

    :param reference_sample: sample of unbinned reference
    :type reference_sample: array_like, n-Dimensional
    :param n_part_x: Number of partitions in x
    :type n_part_x: int
    :param n_part_y: Number of partitions in x, defaults to None
        If None: n_part_y = n_part_x
    :type n_part_y: int, optional
    :param order: Order in which the partitioning is performed, defaults to 'xy'
        'xy' : first bin x then bin y for each partition in x
        'yx' : first bin y then bin x for each partition in y
    :type order: str, optional
    :return: Returns bin_edges_first and bin_edges_second. For order 'xy'('yx')
        these are the bin edges in x(y) and y(x) respectively. bin_edges_second
        is a list of bin edges corresponding to the partitions defined in
        bin_edges_first.
    :rtype: array, 2d array
    :raises ValueError: when an unknown order is passed.

    .. note::
        Reference: F. James, 2008: "Statistical Methods in Experimental
                    Physics", Ch. 11.2.3
    """
    order_ind = get_order_ind(order)  # [0, 1] for 'xy', [1, 0] for 'yx'

    if n_part_y is None:
        n_part_y = n_part_x
    n_bins = [n_part_x, n_part_y]

    # Define data that is binned first and second based on the order argument
    first = np.vstack(reference_sample.T[order_ind[0]])
    second = np.vstack(reference_sample.T[order_ind[1]])

    # Get binning in first dimension:
    enc = KBinsDiscretizer(n_bins=n_bins[order_ind[0]], encode='ordinal',
                           strategy='quantile')
    enc.fit(first)
    bin_edges_first = enc.bin_edges_[0]
    bin_edges_first[0] = -np.inf
    bin_edges_first[-1] = np.inf

    # Get binning in second dimension (for each bin in first dimension):
    enc = KBinsDiscretizer(n_bins=n_bins[order_ind[1]], encode='ordinal',
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

    return bin_edges_first, bin_edges_second


def apply_irregular_binning(data_sample, bin_edges_first, bin_edges_second,
                            order='xy'):
    """Apply irregular binning to data sample.

    :param data_sample: Sample of unbinned data.
    :type data_sample: array
    :param bin_edges_first: Array of bin edges in first dimension
    :type bin_edges_first: array
    :param bin_edges_second: Array of bin edges in second dimension for slices
        in first dimension
    :type bin_edges_second: 2d array
    :param order: Order in which the partitioning is performed, defaults to 'xy'
        'xy' : first bin x then bin y for each partition in x
        'yx' : first bin y then bin x for each partition in y
    :type order: str, optional
    :return: binned data. Number of counts in each bin.
    :rtype: array
    """
    order_ind = get_order_ind(order)
    first = np.vstack(data_sample.T[order_ind[0]])
    second = np.vstack(data_sample.T[order_ind[1]])

    ns = []
    i = 0
    for low, high in zip(bin_edges_first[:-1], bin_edges_first[1:]):
        mask = (first > low) & (first <= high)
        n, _ = np.histogram(second[mask], bins=bin_edges_second[i])
        ns.append(n)
        i += 1
    assert len(data_sample) == np.sum(ns), (f'Sum of binned data {np.sum(ns)}'
                                            + ' unequal to size of data sample'
                                            + ' {len(data_sample)}')
    return np.array(ns)


def plot_irregular_binning(ax, bin_edges_first, bin_edges_second,
                           order='xy', c='k',  # 'mediumvioletred',
                           **kwargs):
    """Plot the bin edges as a grid.

    :param ax: axis to plot to
    :type ax: matplotlib axis
    :param bin_edges_first: Array of bin edges in first dimension
    :type bin_edges_first: array
    :param bin_edges_second: Array of bin edges in second dimension for slices
        in first dimension
    :type bin_edges_second: 2d array
    :param order: Order in which the partitioning is performed, defaults to 'xy'
        'xy' : first bin x then bin y for each partition in x
        'yx' : first bin y then bin x for each partition in y
    :type order: str, optional
    :param c: color of the grid, defaults to 'dodgerblue'
    :type c: str, optional
    :param kwargs: kwargs are passed to the plot functions
    :raises ValueError: when an unknown order is passed.
    """
    ylim = ax.get_ylim()
    xlim = ax.get_xlim()
    be_first = bin_edges_first.copy()
    be_second = bin_edges_second.copy()

    if order == 'xy':
        plot_funcs = [ax.axvline, ax.hlines]
        be_first[bin_edges_first == -np.inf] = xlim[0]
        be_first[bin_edges_first == np.inf] = xlim[1]
        be_second[bin_edges_second == -np.inf] = ylim[0]
        be_second[bin_edges_second == np.inf] = ylim[1]
    elif order == 'yx':
        plot_funcs = [ax.axhline, ax.vlines]
        be_first[bin_edges_first == -np.inf] = ylim[0]
        be_first[bin_edges_first == np.inf] = ylim[1]
        be_second[bin_edges_second == -np.inf] = xlim[0]
        be_second[bin_edges_second == np.inf] = xlim[1]
    else:
        raise ValueError(f'order {order} is not defined.')

    i = 0
    for low, high in zip(be_first[:-1], be_first[1:]):
        if i > 0:
            plot_funcs[0](low, zorder=4, c=c, **kwargs)
        plot_funcs[1](be_second[i][1:-1], low, high, zorder=4, color=c,
                      **kwargs)
        i += 1


def plot_equiprobable_histogram(ax, data_sample, bin_edges_first,
                                bin_edges_second, order, cmap_midpoint=None,
                                **kwargs):
    """Plot 2d histogram of data sample binned according to the passed
    irregular binning.

    :param ax: axis to plot to
    :type ax: matplotlib axis
    :param data_sample: Sample of unbinned data.
    :type data_sample: array
    :param bin_edges_first: Array of bin edges in first dimension
    :type bin_edges_first: array
    :param bin_edges_second: Array of bin edges in second dimension for slices
        in first dimension
    :type bin_edges_second: 2d array
    :param order: Order in which the partitioning is performed, defaults to 'xy'
        'xy' : first bin x then bin y for each partition in x
        'yx' : first bin y then bin x for each partition in y
    :type order: str, optional
    :param cmap_midpoint: midpoint of the colormap (i.e. expectation value), 
        defaults to None
    :type cmap_midpoint: float, optional
    :raises ValueError: when an unknown order is passed.
    """
    ylim = ax.get_ylim()
    xlim = ax.get_xlim()
    be_first = bin_edges_first.copy()
    be_second = bin_edges_second.copy()

    if order == 'xy':
        be_first[bin_edges_first == -np.inf] = xlim[0]
        be_first[bin_edges_first == np.inf] = xlim[1]
        be_second[bin_edges_second == -np.inf] = ylim[0]
        be_second[bin_edges_second == np.inf] = ylim[1]
    elif order == 'yx':
        be_first[bin_edges_first == -np.inf] = ylim[0]
        be_first[bin_edges_first == np.inf] = ylim[1]
        be_second[bin_edges_second == -np.inf] = xlim[0]
        be_second[bin_edges_second == np.inf] = xlim[1]
    else:
        raise ValueError(f'order {order} is not defined.')

    ns = apply_irregular_binning(data_sample, bin_edges_first,
                                 bin_edges_second, order=order)

    # get colormap and norm for colorbar
    cmap = mpl.cm.get_cmap('RdBu').reversed()
    if cmap_midpoint is None:
        norm = mpl.colors.Normalize(vmin=ns.min(), vmax=ns.max())
    else:
        delta = max(cmap_midpoint - ns.min(), ns.max() - cmap_midpoint)
        norm = mpl.colors.Normalize(
            vmin=cmap_midpoint - delta, vmax=cmap_midpoint + delta)

    # plot rectangle for each bin
    i = 0
    for low_f, high_f in zip(be_first[:-1], be_first[1:]):
        j = 0
        for low_s, high_s in zip(be_second[i][:-1], be_second[i][1:]):
            if order == 'xy':
                rec = Rectangle((low_f, low_s),
                                high_f - low_f,
                                high_s - low_s,
                                facecolor=cmap(norm(ns[i][j])),
                                **kwargs)
            elif order == 'yx':
                rec = Rectangle((low_s, low_f),
                                high_s - low_s,
                                high_f - low_f,
                                facecolor=cmap(norm(ns[i][j])),
                                **kwargs)
            ax.add_patch(rec)
            j += 1
        i += 1
    fig = mpl.pyplot.gcf()
    fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax,
                 label='Counts per Bin')


def get_order_ind(order):
    """Get order indices:
        [0, 1] for 'xy', [1, 0] for 'yx'
    """
    if order == 'xy':
        order_ind = [0, 1]
    elif order == 'yx':
        order_ind = [1, 0]
    else:
        raise ValueError(f'order {order} is not defined.')
    return order_ind
