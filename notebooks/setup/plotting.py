"""
This module provides multiple methods to plot model output.
"""
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import pint


def generate_figure(subplot_shape=(1,), figsize=(15, 10), facecolor='w', **kwargs):
    """
    Generate a figure window - effectively the same as calling the 'plt.subplots' method, only with
    some default parameter values.

    Parameters
    ----------
    subplot_shape: 2-element tuple
        Determine the number of panel rows and columns, respectively.
    figsize: 2-element tuple
        Determine the figure's width and height, respectively.
    facecolor:str or 3-element tuple
        Figure background color

    Returns
    -------
    fig: Matplotlib figure handle
    ax: Matplotlib axes handle
    """
    fig, ax = plt.subplots(*subplot_shape, figsize=figsize, facecolor='w', **kwargs)
    return fig, ax


def plot_curtain(ci_model, which_pop=None, field_to_plot="", x=None, y=None, aer_z=None, dim_treat="sum",
                 cmap=None, vmin="auto", vmax="auto", auto_diverging=True,
                 ax=None, colorbar=True, cbar_label=None,
                 xscale=None, yscale=None, log_plot=False, title=None, grid=False, xlabel=None, ylabel=None,
                 tight_layout=True, font_size=None, xtick=None, xticklabel=None, ytick=None, yticklabel=None,
                 xlim=None, ylim=None, **kwargs):
    """
    Generate a curtain plot of an aerosol population or another model field.

    Parameters
    ----------
    ci_model: ci_model class
        Containing model run output.
    which_pop: list, str, or None
        Name of aerosol population to plot. If field_to_plot is "Jhet" (ABIFM), "ns_raw" (singular), or "inp_pct"
        (singular) then plotting the population's Jhet, raw ns parameterization (or estimate using singular), or
        the INP percentage relative to the total initial aerosol concentration (ignoring vertical weighting),
        respectively.
        If a list, then adding all aerosol populations together after checking that the "diam" (ABIFM) or "T"
        (singular) arrays have the same length and values.
        If None, then plot a field from the ci_model.ds xr.Dataset.
    field_to_plot: str
        Name of field to plot. If "Jhet" (ABIFM), "ns_raw" (singular), or "inp_pct" (singular) then remember to
        provide 'which_pop'.
    x: str
        coordinate to place on the x-axis - choose between "time", "height", "diam" (ABIFM), or "T" (singular).
        "time", "height", "diam", and "T" are automatically changed to "time_h", "height_km", "diam_um", and "T_C"
        if these converted units serve as the current 'ci_model' object xr.Dataset's dims (x is None).
    y: str
        coordinate to place on the y-axis - choose between "time", "height", "diam" (ABIFM), or "T" (singular).
        "time", "height", "diam", and "T" are automatically changed to "time_h", "height_km", "diam_um", and "T_C"
        if these converted units serve as the current 'ci_model' object xr.Dataset's dims (y is None).
    aer_z: float, int, 2-element tuple, or None
        Only for plotting of the aerosol field (ndim=3). Use a float to specify a 3rd dim coordinate value to use
        for plotting, int for 3rd coordinate index, tuple of floats to define a range of values (plotting a mean
        of that coordinate range), tuple of ints to define a range of indices (plotting a mean of that coordinate
        indices range), and None to plot mean over the full coordinate range (values should match the dim units).
    dim_treat: str
        Relevant if aer_z is a tuple or None. Use "mean", "sum", or "sd" for mean, sum, or standard deviation,
        respectively.
    cmap: str or matplotlib.cm.cmap
        colormap to use to use
    vmin: str or float
        if "auto" then using 1st percentil. If float, defining minimum colormap value
    vmax: str or float
        if "auto" then using 99th percentil. If float, defining maximum colormap value
    auto_diverging: bool
        if True and vmin and/or vmax are "auto" then setting these two variables such that vmin = -x and
        vmax = x where x is max(abs(1st, 99th)). Applicable only to fields specified in the code under the
        'diverging_vars' list.
    ax: matplotlib axis handle
        axes to plot on. If None, then creating a new figure.
    colorbar: bool
        Create a colorbar if true. Handle is returned in that case.
    cbar_label: str
        colorbar lable. Using the field name if None
    xscale: str or None
        "linear" or "log" scale for x-axis. If None, using "linear" for "time", "height", or "T" (singular)
        and "log" for "diam" (ABIFM).
    yscale: str or None
        "linear" or "log" scale for y-axis. If None, using "linear" for "time", "height", or "T" (singular)
        and "log" for "diam" (ABIFM).
    log_plot: bool
        scale for the c-axis (color-scale). Choose between linear (False) or log-scale (True).
    title: str or None
        panel (subplot) title if str
    grid: bool
        If True, add grid.
    xlabel: str or None
        set xlabel with str. Use field name and units if None
    ylabel: str or None
        set ylabel with str. Use field name and units if None
    tight_layout: bool
        If True, using a tight layout (no text overlap or text cut at figure edges) - typically, the
        desired behavior).
    xtick: list, np.ndarray, or None
        if provided, then used as x-axis ticks (need to match the current dim units).
    xticklabel: list, np.ndarray, or None
        if provided, then used as x-axis tick labels.
    ytick: list, np.ndarray, or None
        if provided, then used as y-axis ticks (need to match the current dim units).
    yticklabel: list, np.ndarray, or None
        if provided, then used as y-axis tick labels.
    xlim: 2-elemnt tuple (or list) or None
        xlim range (need to match the current dim units).
    ylim: 2-elemnt tuple (or list) or None
        ylim range (need to match the current dim units).

    Returns
    -------
    ax: Matplotlib axes handle
    cb: Matplotlib colorbar handle (if 'colorbar' is True).
    """
    if x is None:
        x = ci_model.t_out_dim
    if y is None:
        y = ci_model.height_dim

    aer_pop_aux_fields, aer_pop_w_diams, aer_pop_w_diams_str, diverging_vars = \
        set_out_field_namelists(ci_model, "curtain")
    if which_pop is None:  # plot a field from ci_model.ds
        if field_to_plot in ci_model.ds.keys():
            plot_data = ci_model.ds[field_to_plot].copy()
            xf, yf = ci_model.ds[x], ci_model.ds[y]
        else:
            raise KeyError("Could not find the field: '%s' in ci_model.ds. Check for typos, etc." % field_to_plot)
    elif isinstance(which_pop, (list, str)):
        if isinstance(which_pop, str):
            if np.logical_and(which_pop in ci_model.aer.keys(), field_to_plot in aer_pop_aux_fields):
                if np.logical_and(not ci_model.use_ABIFM, field_to_plot == "Jhet"):
                    raise KeyError("Jhet was requested but use_ABIFM is False. Please check your input!")
                plot_data = ci_model.aer[which_pop].ds[field_to_plot].copy()
                xf, yf = ci_model.ds[x], ci_model.ds[y]
            elif np.logical_and(which_pop in ci_model.aer.keys(), field_to_plot in aer_pop_aux_fields):
                if np.logical_and(ci_model.use_ABIFM, field_to_plot in ["ns_raw"]):
                    raise KeyError("%s was requested but use_ABIFM is True. Please check your input!" %
                                   field_to_plot)
                plot_data = ci_model.aer[which_pop].ds[field_to_plot].copy()
                xf, yf = ci_model.ds[x], ci_model.ds[y]
            elif field_to_plot in ci_model.ds.keys():
                print("NOTE: %s was specified: it is in the model object's dataset but not in the %s population; "
                      "assuming the model object dataset field was requested" % (field_to_plot, which_pop))
                plot_data = ci_model.ds[field_to_plot].copy()
                xf, yf = ci_model.ds[x], ci_model.ds[y]
                which_pop = [None]
            else:
                which_pop = [which_pop]
        if np.logical_and(np.all([x in ci_model.aer.keys() for x in which_pop]),
                          field_to_plot not in aer_pop_aux_fields):
            for ii in range(len(which_pop)):
                if ii == 0:
                    if np.logical_and(field_to_plot in aer_pop_w_diams,
                                      field_to_plot in ci_model.aer[which_pop[ii]].ds.keys()):
                        plot_data = ci_model.aer[which_pop[ii]].ds[field_to_plot].copy()
                        # label = "%s %s" % (which_pop[ii], aer_pop_w_diams_str[field_to_plot])
                    else:  # Default - simply plot n_aer
                        print("NOTE: %s was specified: it is not in the model's dataset nor in the %s population; "
                              "plotting the default field (n_aer)" % (field_to_plot, which_pop))
                        plot_data = ci_model.aer[which_pop[ii]].ds["n_aer"].copy()  # plot aerosol field
                        # label = "%s %s" % (which_pop[ii], aer_pop_w_diams_str["n_aer"])
                    if np.logical_or(ci_model.aer[which_pop[ii]].is_INAS, ci_model.use_ABIFM):
                        aer_dim = ci_model.diam_dim
                    else:
                        aer_dim = ci_model.T_dim
                else:
                    if plot_data[aer_dim].size == ci_model.aer[which_pop[ii]].ds[aer_dim].size:
                        if np.all(plot_data[aer_dim].values == ci_model.aer[which_pop[ii]].ds[aer_dim].values):
                            plot_data += ci_model.aer[which_pop[ii]].ds["n_aer"]
                        else:
                            raise RuntimeError("different aerosol %s dim - %s arrays must have the same length \
                                               and values (interpolation might be added updated in future \
                                               updates)" % (aer_dim, aer_dim))
                    elif not ci_model.aer[which_pop[ii]].is_INAS:
                        plot_data += ci_model.aer[which_pop[ii]].ds["n_aer"]
                    else:
                        raise RuntimeError("different aerosol %s dim - %s arrays must have the same length \
                                           and values (interpolation might be added updated in future updates)"
                                           % (aer_dim, aer_dim))
            xf, yf = plot_data[x], plot_data[y]
            if np.logical_or(ci_model.aer[which_pop[ii]].is_INAS, ci_model.use_ABIFM):
                possible_fields = {ci_model.height_dim, ci_model.t_out_dim, ci_model.diam_dim}
            else:
                possible_fields = {ci_model.height_dim, ci_model.t_out_dim, ci_model.T_dim}
            [possible_fields.remove(fn) for fn in [x, y] if fn in possible_fields]
            if len(possible_fields) > 1:
                raise RuntimeError("something is not right - too many optional fields \
                                   (check 'x' and 'y' string values)")
            z = possible_fields.pop()
            if np.logical_or(np.logical_or(ci_model.aer[which_pop[ii]].is_INAS, ci_model.use_ABIFM),
                             np.logical_and(field_to_plot == "inp", not ci_model.aer[which_pop[ii]].is_INAS)):
                plot_data = process_dim(plot_data, z, aer_z, dim_treat)
        elif np.logical_and(field_to_plot not in aer_pop_aux_fields, not which_pop[0] is None):
            raise KeyError("Could not find one or more of the requested aerosl population names: \
                           '%s' in ci_model.aer. Check for typos, etc." % which_pop)

    # remove pint.Quantity data
    if isinstance(plot_data.data, pint.Quantity):
        plot_data.data = plot_data.data.magnitude

    # arrange plot dims
    if x == plot_data.dims[0]:
        plot_data = plot_data.transpose()

    if cmap is None:
        if np.logical_and(auto_diverging, field_to_plot in diverging_vars):
            cmap = "bwr"
        else:
            cmap = "cubehelix"

    if np.logical_or(vmin == "auto", vmax == "auto"):
        if vmin == "auto":
            vmin = np.percentile(plot_data, 1)
        if vmax == "auto":
            vmax = np.percentile(plot_data, 99)
        if np.logical_and(auto_diverging, field_to_plot in diverging_vars):
            vmin = np.max(np.abs([vmin, vmax])) * (-1)
            vmax = np.max(np.abs([vmin, vmax]))

    if xlabel is None:
        if "units" in plot_data[x].attrs:
            xlabel = "%s [%s]" % (x, plot_data[x].attrs["units"])
        else:
            xlabel = "%s" % x
    if ylabel is None:
        if "units" in plot_data[y].attrs:
            ylabel = "%s [%s]" % (y, plot_data[y].attrs["units"])
        else:
            ylabel = "%s" % y

    if np.logical_and(xscale is None, x == ci_model.diam_dim):
        xscale = "log"
    if np.logical_and(yscale is None, y == ci_model.diam_dim):
        yscale = "log"

    def corners(array):
        data = array.values
        delta = data[1] - data[0]
        return np.concatenate(((data[0] - delta / 2,), data + delta / 2))

    if log_plot is True:
        mesh = ax.pcolormesh(corners(xf), corners(yf), plot_data, norm=colors.LogNorm(vmin=vmin, vmax=vmax),
                             cmap=cmap, **kwargs)
    else:
        mesh = ax.pcolormesh(corners(xf), corners(yf), plot_data, cmap=cmap,
                             vmin=vmin, vmax=vmax, **kwargs)

    fine_tuning(ax, title, xscale, yscale, grid, xlabel, ylabel, tight_layout, font_size, xtick, xticklabel,
                ytick, yticklabel, xlim, ylim)

    if colorbar:
        cb = plt.colorbar(mesh, ax=ax)
        if cbar_label is None:
            if "units" in plot_data.attrs:
                cb.set_label("%s" % plot_data.attrs["units"])
        else:
            cb.set_label(cbar_label)
        if font_size is not None:
            cb.ax.tick_params(labelsize=font_size)
            cb.ax.set_ylabel(cb.ax.get_yaxis().label.get_text(), fontsize=font_size)
        return ax, cb

    return ax


def plot_tseries(ci_model, which_pop=None, field_to_plot="", aer_z=None, dim_treat="sum",
                 Height=None, Height_dim_treat="mean", ax=None,
                 yscale=None, title=None, grid=False, xlabel=None, ylabel=None, tight_layout=True,
                 font_size=16, xtick=None, xticklabel=None, ytick=None, yticklabel=None, legend=None,
                 xlim=None, ylim=None, legend_label=None, **kwargs):
    """
    Generates aerosol or other model output field's time series

    Parameters
    ----------
    ci_model: ci_model class
        Containing model run output.
    which_pop: str or None
        Name of aerosol population to plot. If field_to_plot is "Jhet" (ABIFM), "ns_raw" (singular), or "inp_pct"
        (singular) then plotting the population's Jhet, raw ns parameterization (or estimate using singular), or
        the INP percentage relative to the total initial aerosol concentration (ignoring vertical weighting),
        respectively.
        If None, then plot a field from the ci_model.ds xr.Dataset.
    field_to_plot: str
        Name of field to plot. If "Jhet" (ABIFM), "ns_raw" (singular), or "inp_pct" (singular) then remember to
        provide 'which_pop'.
    aer_z: float, int, 2-element tuple, or None
        Only for plotting of the aerosol field (ndim=3). Use a float to specify a 3rd dim coordinate value to use
        for plotting, int for 3rd coordinate index, tuple of floats to define a range of values (plotting a mean
        of that coordinate range), tuple of ints to define a range of indices (plotting a mean of that coordinate
        indices range), and None to plot mean over the full coordinate range (values should match the dim units).
    dim_treat: str
        Relevant if aer_z is a tuple or None. Use "mean", "sum", or "sd" for mean, sum, or standard deviation,
        respectively.
    Height: float, int, 2-element tuple, list, or None
        Height elements, range, or values to treat from the dim. Options:
        1. float to specify a dim coordinate value to use for plotting (need to match the current dim units).
        2. int for coordinate index.
        3. tuple of floats to define a range of values (need to match the current dim units).
        4. tuple of ints to define a range of indices.
        5. list or np.ndarray of floats to define a specific values (need to match the current dim units).
        6. list or np.ndarray of ints to define a specific indices.
        7. None to take the full coordinate range.
    Height_dim_treat: str
        How to treat the height dimension. Use "mean", "sum", "sd", None for mean, sum, standard deviation,
        non-treatment (keep dim),
        respectively.
    ax: matplotlib axis handle
        axes to plot on. If None, then creating a new figure.
    yscale: str or None
        "linear" or "log" scale for y-axis. If None, using "linear" for "time", "height", or "T" (singular)
        and "log" for "diam" (ABIFM).
    title: str or None
        panel (subplot) title if str
    grid: bool
        If True, add grid.
    xlabel: str or None
        set xlabel with str. Use field name and units if None
    ylabel: str or None
        set ylabel with str. Use field name and units if None
    tight_layout: bool
        If True, using a tight layout (no text overlap or text cut at figure edges) - typically, the
        desired behavior).
    xtick: list, np.ndarray, or None
        if provided, then used as x-axis ticks (need to match the current dim units).
    xticklabel: list, np.ndarray, or None
        if provided, then used as x-axis tick labels.
    ytick: list, np.ndarray, or None
        if provided, then used as y-axis ticks (need to match the current dim units).
    yticklabel: list, np.ndarray, or None
        if provided, then used as y-axis tick labels.
    legend: bool or None
        if None, placing legend if processed plot_data has ndim = 2.
    xlim: 2-elemnt tuple (or list) or None
        xlim range (need to match the current dim units).
    ylim: 2-elemnt tuple (or list) or None
        ylim range (need to match the current dim units).
    legend_label: str or None:
        Label to set for the legend (only valid if plotting a single curve). Using default if None.

    Returns
    -------
    ax: Matplotlib axes handle
    """
    aer_pop_aux_fields, aer_pop_w_diams, aer_pop_w_diams_str = set_out_field_namelists(ci_model, "tseries")
    if which_pop is None:  # plot a field from ci_model.ds
        if field_to_plot in ci_model.ds.keys():
            plot_data = ci_model.ds[field_to_plot].copy()
            label = field_to_plot
        else:
            raise KeyError("Could not find the field: '%s' in ci_model.ds. Check for typos, etc." % field_to_plot)
    elif isinstance(which_pop, (list, str)):
        if isinstance(which_pop, str):
            if np.logical_and(which_pop in ci_model.aer.keys(), field_to_plot in aer_pop_aux_fields):
                if np.logical_and(not ci_model.use_ABIFM, field_to_plot == "Jhet"):
                    raise KeyError("Jhet was requested but use_ABIFM is False. Please check your input!")
                plot_data = ci_model.aer[which_pop].ds[field_to_plot].copy()
                label = "%s %s" % (which_pop, field_to_plot)
            elif np.logical_and(which_pop in ci_model.aer.keys(), field_to_plot in aer_pop_aux_fields):
                if np.logical_and(ci_model.use_ABIFM, field_to_plot in ["ns_raw"]):
                    raise KeyError("%s was requested but use_ABIFM is True. Please check your input!" %
                                   field_to_plot)
                plot_data = ci_model.aer[which_pop].ds[field_to_plot].copy()
                label = "%s %s" % (which_pop, field_to_plot)
            elif field_to_plot in ci_model.ds.keys():
                print("NOTE: %s was specified: it is in the model object's dataset but not in the %s population; "
                      "assuming the model object dataset field was requested" % (field_to_plot, which_pop))
                plot_data = ci_model.ds[field_to_plot].copy()
                which_pop = [None]
            else:
                which_pop = [which_pop]
        if np.logical_and(np.all([x in ci_model.aer.keys() for x in which_pop]),
                          field_to_plot not in aer_pop_aux_fields):
            for ii in range(len(which_pop)):
                if ii == 0:
                    if np.logical_and(field_to_plot in aer_pop_w_diams,
                                      field_to_plot in ci_model.aer[which_pop[ii]].ds.keys()):
                        plot_data = ci_model.aer[which_pop[ii]].ds[field_to_plot].copy()
                        label = "%s %s" % (which_pop[ii], aer_pop_w_diams_str[field_to_plot])
                    else:  # Default - simply plot n_aer
                        print("NOTE: %s was specified: it is not in the model's dataset nor in the %s population; "
                              "plotting the default field (n_aer)" % (field_to_plot, which_pop))
                        plot_data = ci_model.aer[which_pop[ii]].ds["n_aer"].copy()  # plot aerosol field
                        label = "%s %s" % (which_pop[ii], aer_pop_w_diams_str["n_aer"])
                    if np.logical_or(ci_model.aer[which_pop[ii]].is_INAS, ci_model.use_ABIFM):
                        aer_dim = ci_model.diam_dim
                    else:
                        aer_dim = ci_model.T_dim
                else:
                    if plot_data[aer_dim].size == ci_model.aer[which_pop[ii]].ds[aer_dim].size:
                        if np.all(plot_data[aer_dim].values == ci_model.aer[which_pop[ii]].ds[aer_dim].values):
                            plot_data += ci_model.aer[which_pop[ii]].ds["n_aer"]
                        else:
                            raise RuntimeError("different aerosol %s dim - %s arrays must have the same length \
                                               and values (interpolation might be added updated in future \
                                               updates)" % (aer_dim, aer_dim))
                    elif not ci_model.aer[which_pop[ii]].is_INAS:
                        plot_data += ci_model.aer[which_pop[ii]].ds["n_aer"]
                    else:
                        raise RuntimeError("different aerosol %s dim - %s arrays must have the same length \
                                           and values (interpolation might be added updated in future updates)"
                                           % (aer_dim, aer_dim))
            if np.logical_or(np.logical_or(ci_model.aer[which_pop[ii]].is_INAS, ci_model.use_ABIFM),
                             np.logical_and(field_to_plot == "inp", not ci_model.aer[which_pop[ii]].is_INAS)):
                plot_data = process_dim(plot_data, aer_dim, aer_z, dim_treat)
        elif np.logical_and(field_to_plot not in aer_pop_aux_fields, not which_pop[0] is None):
            raise KeyError("Could not find one or more of the requested aerosl population names: \
                           '%s' in ci_model.aer. Check for typos, etc." % which_pop)

    # remove pint.Quantity data
    if isinstance(plot_data.data, pint.Quantity):
        plot_data.data = plot_data.data.magnitude

    # Select values or indices from the height dim and treat (mean, sum, as-is).
    if plot_data.ndim == 2:
        plot_data = process_dim(plot_data, ci_model.height_dim, Height, Height_dim_treat)

    if xlabel is None:
        if "units" in plot_data[ci_model.t_out_dim].attrs:
            xlabel = "%s [%s]" % ("time", plot_data[ci_model.t_out_dim].attrs["units"])
        else:
            xlabel = "time"
    if ylabel is None:
        if "units" in plot_data.attrs:
            ylabel = "%s [%s]" % (label, plot_data.attrs["units"])
        else:
            ylabel = label

    if plot_data.ndim == 3:
        raise RuntimeError("processed aerosol field still has 3 dimensions. Consider reducing by selecting \
                           a single values or indices or average/sum")
    elif plot_data.ndim == 2:
        dim_2nd = [x for x in plot_data.dims if x != ci_model.t_out_dim][0]  # dim for loop (height unless aerosol)
        for ii in range(plot_data[dim_2nd].size):
            if "units" in plot_data[dim_2nd].attrs:
                label_p = label + " (%s = %.1f %s)" % (dim_2nd, plot_data[dim_2nd][ii],
                                                       plot_data[dim_2nd].attrs["units"])
            else:
                label_p = label + " (%s = %.1f)" % (dim_2nd, plot_data[dim_2nd][ii])
            if legend_label is not None:  # Assuming a string
                label_p = legend_label
            ax.plot(plot_data[ci_model.t_out_dim], plot_data.isel({dim_2nd: ii}), label=label_p, **kwargs)
        if legend is None:
            legend = True
    else:
        if legend_label is not None:  # Assuming a string
            label = legend_label
        ax.plot(plot_data[ci_model.t_out_dim], plot_data, label=label, **kwargs)

    ax = fine_tuning(ax, title, None, yscale, grid, xlabel, ylabel, tight_layout, font_size, xtick, xticklabel,
                     ytick, yticklabel, xlim, ylim)

    if legend is True:
        ax.legend()

    return ax


def plot_profile(ci_model, which_pop=None, field_to_plot="", aer_z=None, dim_treat="sum",
                 Time=None, Time_dim_treat="mean", ax=None,
                 xscale=None, title=None, grid=False, xlabel=None, ylabel=None, tight_layout=True,
                 font_size=16, xtick=None, xticklabel=None, ytick=None, yticklabel=None, legend=None,
                 xlim=None, ylim=None, legend_label=None, cld_bnd=False, **kwargs):

    """
    Generates aerosol population or other model output field's profile

    Parameters
    ----------
    ci_model: ci_model class
        Containing model run output.
    which_pop: str or None
        Name of aerosol population to plot. If field_to_plot is "Jhet" (ABIFM), "ns_raw" (singular), or "inp_pct"
        (singular) then plotting the population's Jhet, raw ns parameterization (or estimate using singular), or
        the INP percentage relative to the total initial aerosol concentration (ignoring vertical weighting),
        respectively. If None, then plot a field from the ci_model.ds xr.Dataset.
        Name of field to plot.
        If "Jhet" (ABIFM), "ns_raw" (singular), or "inp_pct" (singular) then remember to
        provide 'which_pop'.
    aer_z: float, int, 2-element tuple, or None
        Only for plotting of the n_aer field (ndim=3). Use a float to specify a 3rd dim coordinate value to use
        for plotting, int for 3rd coordinate index, tuple of floats to define a range of values (plotting a mean
        of that coordinate range), tuple of ints to define a range of indices (plotting a mean of that coordinate
        indices range), and None to plot mean over the full coordinate range (values should match the dim units).
    dim_treat: str
        Relevant if aer_z is a tuple or None. Use "mean", "sum", or "sd" for mean, sum, or standard deviation,
        respectively.
    Time: float, int, 2-element tuple, list, or None
        Time elements, range, or values to treat from the dim. Options:
        1. float to specify a dim coordinate value to use for plotting (need to match the current dim units).
        2. int for coordinate index.
        3. tuple of floats to define a range of values (need to match the current dim units).
        4. tuple of ints to define a range of indices.
        5. list or np.ndarray of floats to define a specific values (need to match the current dim units).
        6. list or np.ndarray of ints to define a specific indices.
        7. None to take the full coordinate range.
    Time_dim_treat: str
        How to treat the time dimension. Use "mean", "sum", "sd", None for mean, sum, standard deviation,
        non-treatment (keep dim),
        respectively.
    ax: matplotlib axis handle
        axes to plot on. If None, then creating a new figure.
    xscale: str or None
        "linear" or "log" scale for x-axis. If None, using "linear" for "time", "height", or "T" (singular)
        and "log" for "diam" (ABIFM).
    title: str or None
        panel (subplot) title if str
    grid: bool
        If True, add grid.
    xlabel: str or None
        set xlabel with str. Use field name and units if None
    ylabel: str or None
        set ylabel with str. Use field name and units if None
    tight_layout: bool
        If True, using a tight layout (no text overlap or text cut at figure edges) - typically, the
        desired behavior).
    xtick: list, np.ndarray, or None
        if provided, then used as x-axis ticks (need to match the current dim units).
    xticklabel: list, np.ndarray, or None
        if provided, then used as x-axis tick labels.
    ytick: list, np.ndarray, or None
        if provided, then used as y-axis ticks (need to match the current dim units).
    yticklabel: list, np.ndarray, or None
        if provided, then used as y-axis tick labels.
    legend: bool or None
        if None, placing legend if processed plot_data has ndim = 2.
    xlim: 2-elemnt tuple (or list) or None
        xlim range (need to match the current dim units).
    ylim: 2-elemnt tuple (or list) or None
        ylim range (need to match the current dim units).
    legend_label: str or None:
        Label to set for the legend (only valid if plotting a single curve). Using default if None.
    cld_bnd: bool or dict
        if True, plotting cloud boundary patches and/or lines. If dict, then can include the keys 'p_color',
        'p_alpha', 'l_color', 'l_width', 'l_style', and 'x_rng' for cloud boundaries' patch color and transparency,
        and boundary line color, width, and style. If a dict is provided with None key values, default parameters
        are used. 'x_rng' determines the x-axis values (min and max values) for the patch and/or line
        set 'p_color' to None to avoid generating a patch and 'l_color' to None to avoid plotting boundaries.

    Returns
    -------
    ax: Matplotlib axes handle
    """
    aer_pop_aux_fields, aer_pop_w_diams, aer_pop_w_diams_str = set_out_field_namelists(ci_model, "profile")
    if which_pop is None:  # plot a field from ci_model.ds
        if field_to_plot in ci_model.ds.keys():
            plot_data = ci_model.ds[field_to_plot].copy()
            label = field_to_plot
        else:
            raise KeyError("Could not find the field: '%s' in ci_model.ds. Check for typos, etc." % field_to_plot)
    elif isinstance(which_pop, (list, str)):
        if isinstance(which_pop, str):
            if np.logical_and(which_pop in ci_model.aer.keys(), field_to_plot in aer_pop_aux_fields):
                if np.logical_and(not ci_model.use_ABIFM, field_to_plot == "Jhet"):
                    raise KeyError("Jhet for was requested but use_ABIFM is False. Please check your input!")
                plot_data = ci_model.aer[which_pop].ds[field_to_plot].copy()
                label = "%s %s" % (which_pop, field_to_plot)
            elif np.logical_and(which_pop in ci_model.aer.keys(), field_to_plot in aer_pop_aux_fields):
                if np.logical_and(ci_model.use_ABIFM, field_to_plot in ["ns_raw"]):
                    raise KeyError("%s was requested but use_ABIFM is True. Please check your input!" %
                                   field_to_plot)
                plot_data = ci_model.aer[which_pop].ds[field_to_plot].copy()
                label = "%s %s" % (which_pop, field_to_plot)
            elif field_to_plot in ci_model.ds.keys():
                print("NOTE: %s was specified: it is in the model object's dataset but not in the %s population; "
                      "assuming the model object dataset field was requested" % (field_to_plot, which_pop))
                plot_data = ci_model.ds[field_to_plot].copy()
                which_pop = [None]
            else:
                which_pop = [which_pop]
        if np.logical_and(np.all([x in ci_model.aer.keys() for x in which_pop]),
                          field_to_plot not in aer_pop_aux_fields):
            for ii in range(len(which_pop)):
                if ii == 0:
                    if np.logical_and(field_to_plot in aer_pop_w_diams,
                                      field_to_plot in ci_model.aer[which_pop[ii]].ds.keys()):
                        plot_data = ci_model.aer[which_pop[ii]].ds[field_to_plot].copy()
                        label = "%s %s" % (which_pop[ii], aer_pop_w_diams_str[field_to_plot])
                    else:  # Default - simply plot n_aer
                        print("NOTE: %s was specified: it is not in the model's dataset nor in the %s population; "
                              "plotting the default field (n_aer)" % (field_to_plot, which_pop))
                        plot_data = ci_model.aer[which_pop[ii]].ds["n_aer"].copy()  # plot aerosol field
                        label = "%s %s" % (which_pop[ii], aer_pop_w_diams_str["n_aer"])
                    if np.logical_or(ci_model.aer[which_pop[ii]].is_INAS, ci_model.use_ABIFM):
                        aer_dim = ci_model.diam_dim
                    else:
                        aer_dim = ci_model.T_dim
                else:
                    if plot_data[aer_dim].size == ci_model.aer[which_pop[ii]].ds[aer_dim].size:
                        if np.all(plot_data[aer_dim].values == ci_model.aer[which_pop[ii]].ds[aer_dim].values):
                            plot_data += ci_model.aer[which_pop[ii]].ds["n_aer"]
                        else:
                            raise RuntimeError("different aerosol %s dim - %s arrays must have the same length \
                                               and values (interpolation might be added updated in future \
                                               updates)" % (aer_dim, aer_dim))
                    elif not ci_model.aer[which_pop[ii]].is_INAS:
                        plot_data += ci_model.aer[which_pop[ii]].ds["n_aer"]
                    else:
                        raise RuntimeError("different aerosol %s dim - %s arrays must have the same length \
                                           and values (interpolation might be added updated in future updates)"
                                           % (aer_dim, aer_dim))
            if np.logical_or(np.logical_or(ci_model.aer[which_pop[ii]].is_INAS, ci_model.use_ABIFM),
                             np.logical_and(field_to_plot == "inp", not ci_model.aer[which_pop[ii]].is_INAS)):
                plot_data = process_dim(plot_data, aer_dim, aer_z, dim_treat)
        elif np.logical_and(field_to_plot not in aer_pop_aux_fields, not which_pop[0] is None):
            raise KeyError("Could not find one or more of the requested aerosl population names: \
                           '%s' in ci_model.aer. Check for typos, etc." % which_pop)

    # remove pint.Quantity data
    if isinstance(plot_data.data, pint.Quantity):
        plot_data.data = plot_data.data.magnitude

    # Select values or indices from the time dim and treat (mean, sum, as-is).
    if plot_data.ndim == 2:
        plot_data = process_dim(plot_data, ci_model.t_out_dim, Time, Time_dim_treat)

    if ylabel is None:
        if "units" in plot_data[ci_model.height_dim].attrs:
            ylabel = "%s [%s]" % ("height", plot_data[ci_model.height_dim].attrs["units"])
        else:
            ylabel = "height"
    if xlabel is None:
        if "units" in plot_data.attrs:
            xlabel = "%s [%s]" % (label, plot_data.attrs["units"])
        else:
            xlabel = label

    if isinstance(cld_bnd, (bool, dict)):
        if isinstance(cld_bnd, bool):
            if cld_bnd:
                cld_bnd = {}  # setting a dict - values are added below.
            else:
                cld_bnd = None
        if isinstance(cld_bnd, dict):
            if 'p_color' not in cld_bnd.keys():
                cld_bnd['p_color'] = 'k'
            if 'p_alpha' not in cld_bnd.keys():
                cld_bnd['p_alpha'] = 0.3
            if 'l_color' not in cld_bnd.keys():
                cld_bnd['l_color'] = 'k'
            if 'l_width' not in cld_bnd.keys():
                cld_bnd['l_width'] = 1
            if 'l_style' not in cld_bnd.keys():
                cld_bnd['l_style'] = '-'
            if 'x_rng' not in cld_bnd.keys():
                if xlim is not None:
                    cld_bnd['x_rng'] = xlim
                else:
                    cld_bnd['x_rng'] = [plot_data.min(), plot_data.max()]

    if cld_bnd is not None:
        bounds = [process_dim(ci_model.ds["lowest_cbh"].copy(), ci_model.time_dim, Time, Time_dim_treat),
                  process_dim(ci_model.ds["lowest_cth"].copy(), ci_model.time_dim, Time, Time_dim_treat)]
        if np.logical_and(bounds[0].size == 1, cld_bnd['p_color'] is not None):  # patch only for a single t-step
            ax.fill_between(cld_bnd['x_rng'], bounds[0], bounds[1], label='_nolegend_',
                            color=cld_bnd['p_color'], alpha=cld_bnd['p_alpha'])

    if plot_data.ndim == 3:
        raise RuntimeError("processed aerosol field still had 3 dimensions. Consider reducing by selecting \
                           a single values or indices or average/sum")
    elif plot_data.ndim == 2:
        dim_2nd = [x for x in plot_data.dims if x != ci_model.height_dim][0]  # dim to loop (time unless aerosol).
        for ii in range(plot_data[dim_2nd].size):
            if "units" in plot_data[dim_2nd].attrs:
                label_p = label + " (%s = %.1f %s)" % (dim_2nd, plot_data[dim_2nd][ii],
                                                       plot_data[dim_2nd].attrs["units"])
            else:
                label_p = label + " (%s = %.1f)" % (dim_2nd, plot_data[dim_2nd][ii])
            if legend_label is not None:  # Assuming a string
                label_p = legend_label
            ax.plot(plot_data.isel({dim_2nd: ii}), plot_data[ci_model.height_dim], label=label_p, **kwargs)
            if cld_bnd is not None:
                if cld_bnd['l_color'] is not None:
                    for cldii in range(2):
                        ax.plot(cld_bnd['x_rng'],
                                np.tile(bounds[cldii][ii], (2)), label='_nolegend_', color=cld_bnd['l_color'],
                                linewidth=cld_bnd['l_width'], linestyle=cld_bnd['l_style'])

        if legend is None:
            legend = True
    else:
        if legend_label is not None:  # Assuming a string
            label = legend_label
        ax.plot(plot_data, plot_data[ci_model.height_dim], label=label, **kwargs)
        if cld_bnd is not None:
            if cld_bnd['l_color'] is not None:
                for cldii in range(2):
                    ax.plot(cld_bnd['x_rng'], np.tile(bounds[cldii], (2)), label='_nolegend_',
                            color=cld_bnd['l_color'], linewidth=cld_bnd['l_width'], linestyle=cld_bnd['l_style'])

    ax = fine_tuning(ax, title, xscale, None, grid, xlabel, ylabel, tight_layout, font_size, xtick, xticklabel,
                     ytick, yticklabel, xlim, ylim)

    if legend is True:
        ax.legend()

    return ax


def plot_psd(ci_model, which_pop=None, field_to_plot="",
             Time=None, Time_dim_treat=None, Height=None, Height_dim_treat=None, ax=None,
             xscale=None, yscale=None, title=None, grid=False, xlabel=None, ylabel=None, tight_layout=True,
             font_size=16, xtick=None, xticklabel=None, ytick=None, yticklabel=None, legend=None,
             xlim=None, ylim=None, **kwargs):

    """
    Generates an aerosol population PSD plots

    Parameters
    ----------
    ci_model: ci_model class
        Containing model run output.
    field_to_plot: str
        Name of field to plot. This input parameter can only have an effect in INAS (if "inp_tot" is
        requested).
    which_pop: str or None
        Name of aerosol population to plot.
    Time: float, int, 2-element tuple, list, or None
        Time elements, range, or values to treat from the dim. Options:
        1. float to specify a dim coordinate value to use for plotting  (need to match the current dim units).
        2. int for coordinate index.
        3. tuple of floats to define a range of values (need to match the current dim units).
        4. tuple of ints to define a range of indices.
        5. list or np.ndarray of floats to define a specific values (need to match the current dim units).
        6. list or np.ndarray of ints to define a specific indices.
        7. None to take the full coordinate range.
    Time_dim_treat: str
        How to treat the time dimension. Use "mean", "sum", "sd", None for mean, sum, standard deviation,
        non-treatment (keep dim), respectively.
    Height: float, int, 2-element tuple, list, or None
        Height elements, range, or values to treat from the dim. Options:
        1. float to specify a dim coordinate value to use for plotting (need to match the current dim units).
        2. int for coordinate index.
        3. tuple of floats to define a range of values (need to match the current dim units).
        4. tuple of ints to define a range of indices.
        5. list or np.ndarray of floats to define a specific values (need to match the current dim units).
        6. list or np.ndarray of ints to define a specific indices.
        7. None to take the full coordinate range.
    Height_dim_treat: str
        How to treat the height dimension. Use "mean", "sum", "sd", None for mean, sum, standard deviation,
        non-treatment (keep dim), respectively.
    ax: matplotlib axis handle
        axes to plot on. If None, then creating a new figure.
    xscale: str or None
        "linear" or "log" scale for x-axis. If None, using "linear" for "time", "height", or "T" (singular)
        and "log" for "diam" (ABIFM).
    yscale: str or None
        "linear" or "log" scale for y-axis. If None, using "linear" for "time", "height", or "T" (singular)
        and "log" for "diam" (ABIFM).
    title: str or None
        panel (subplot) title if str
    grid: bool
        If True, add grid.
    xlabel: str or None
        set xlabel with str. Use field name and units if None
    ylabel: str or None
        set ylabel with str. Use field name and units if None
    tight_layout: bool
        If True, using a tight layout (no text overlap or text cut at figure edges) - typically, the
        desired behavior).
    xtick: list, np.ndarray, or None
        if provided, then used as x-axis ticks (check that values match the current dim units).
    xticklabel: list, np.ndarray, or None
        if provided, then used as x-axis tick labels.
    ytick: list, np.ndarray, or None
        if provided, then used as y-axis ticks (check that values match the current dim units).
    yticklabel: list, np.ndarray, or None
        if provided, then used as y-axis tick labels.
    xlim: 2-elemnt tuple (or list) or None
        xlim range (check that values match the current dim units).
    ylim: 2-elemnt tuple (or list) or None
        ylim range (check that values match the current dim units).
    legend: bool or None
        if None, placing legend if processed plot_data has ndim >= 2.

    Returns
    -------
    ax: Matplotlib axes handle
    """
    aer_pop_w_diams, aer_pop_w_diams_str = set_out_field_namelists(ci_model, "psd")
    if isinstance(which_pop, str):
        which_pop = [which_pop]
    if np.all([x in ci_model.aer.keys() for x in which_pop]):
        for ii in range(len(which_pop)):
            if ii == 0:
                if np.logical_or(ci_model.aer[which_pop[ii]].is_INAS, ci_model.use_ABIFM):
                    if field_to_plot == "inp_tot":
                        label = "%s %s" % (which_pop[ii], "INP conc.")
                        plot_data = ci_model.aer[which_pop[ii]].ds["inp_tot"].copy()  # plot INP subset field
                    else:
                        label = "%s %s" % (which_pop[ii], "conc.")
                        plot_data = ci_model.aer[which_pop[ii]].ds["n_aer"].copy()  # plot aerosol field
                    aer_dim = ci_model.diam_dim
                else:
                    label = "%s %s" % (which_pop[ii], "INP T spec.")
                    plot_data = ci_model.aer[which_pop[ii]].ds["inp_snap"].copy()
                    aer_dim = ci_model.T_dim
                if np.logical_and(field_to_plot in aer_pop_w_diams,
                                  field_to_plot in ci_model.aer[which_pop[ii]].ds.keys()):
                    plot_data = ci_model.aer[which_pop[ii]].ds[field_to_plot].copy()
                    label = "%s %s" % (which_pop[ii], aer_pop_w_diams_str[field_to_plot])
                else:  # Default - simply plot n_aer
                    print("NOTE: %s was specified: it is in the model's dataset nor in the %s population; "
                          "assuming the model object dataset field was requested" % (field_to_plot, which_pop))
                    plot_data = ci_model.aer[which_pop[ii]].ds["n_aer"].copy()  # plot aerosol field
                    label = "%s %s" % (which_pop[ii], aer_pop_w_diams_str["n_aer"])
                if np.logical_or(ci_model.aer[which_pop[ii]].is_INAS, ci_model.use_ABIFM):
                    aer_dim = ci_model.diam_dim
                else:
                    aer_dim = ci_model.T_dim
            else:
                if plot_data[aer_dim].size == ci_model.aer[which_pop[ii]].ds[aer_dim].size:
                    if np.all(plot_data[aer_dim].values == ci_model.aer[which_pop[ii]].ds[aer_dim].values):
                        plot_data += ci_model.aer[which_pop[ii]].ds["n_aer"]
                    else:
                        raise RuntimeError("different aerosol %s dim - %s arrays must have the same length \
                                           and values (interpolation might be added updated in future \
                                           updates)" % (aer_dim, aer_dim))
                else:
                    raise RuntimeError("different aerosol %s dim - %s arrays must have the same length \
                                       and values (interpolation might be added updated in future updates)"
                                       % (aer_dim, aer_dim))
    else:
        raise KeyError("Could not find one or more of the requested aerosl population names: \
                       '%s' in ci_model.aer. Check for typos, etc." % which_pop)

    # remove pint.Quantity data
    if isinstance(plot_data.data, pint.Quantity):
        plot_data.data = plot_data.data.magnitude

    # Select values or indices from the time and height dims and treat (mean, sum, as-is).
    plot_data = process_dim(plot_data, ci_model.t_out_dim, Time, Time_dim_treat)
    plot_data = process_dim(plot_data, ci_model.height_dim, Height, Height_dim_treat)

    if xlabel is None:
        xlabel = "%s [%s]" % ("Diameter", plot_data[ci_model.diam_dim].attrs["units"])
    if ylabel is None:
        if "units" in plot_data.attrs:
            ylabel = "%s [%s]" % (label, plot_data.attrs["units"])
        else:
            ylabel = label

    if plot_data.ndim == 3:
        plot_data = plot_data.stack(h_t=(ci_model.height_dim, ci_model.t_out_dim))
        heights = plot_data[ci_model.height_dim].values
        times = plot_data[ci_model.t_out_dim].values
    elif plot_data.ndim == 2:
        if "time" in plot_data.dims:
            times = plot_data[ci_model.t_out_dim].values
            heights = None
            plot_data = plot_data.rename({ci_model.t_out_dim: "h_t"})
        else:
            times = None
            heights = plot_data[ci_model.height_dim].values
            plot_data = plot_data.rename({ci_model.height_dim: "h_t"})
    else:
        heights = None
        times = None
        plot_data = plot_data.expand_dims("h_t")

    for ii in range(plot_data["h_t"].size):
        label_p = label
        if heights is not None:
            label_p = label_p + "; $h$ = %.0f %s" % (heights[ii], ci_model.ds[ci_model.height_dim].attrs["units"])
        if times is not None:
            label_p = label_p + "; $t$ = %.0f %s" % (times[ii], ci_model.ds[ci_model.t_out_dim].attrs["units"])

        ax.plot(plot_data[aer_dim], plot_data.isel({"h_t": ii}).values, label=label_p, **kwargs)
    if legend is None:
        legend = True

    ax = fine_tuning(ax, title, xscale, yscale, grid, xlabel, ylabel, tight_layout, font_size, xtick, xticklabel,
                     ytick, yticklabel, xlim, ylim)

    if legend is True:
        ax.legend()

    return ax


def fine_tuning(ax, title=None, xscale=None, yscale=None, grid=False, xlabel=None, ylabel=None,
                tight_layout=True, font_size=None, xtick=None, xticklabel=None, ytick=None,
                yticklabel=None, xlim=None, ylim=None):
    """
    Fine tune plot (labels, grids, etc.).

    Parameters
    ----------
    ax: matplotlib axis handle
        axes to plot on. If None, then creating a new figure.
    title: str or None
        panel (subplot) title if str
    xscale: str or None
        "linear" or "log" scale for x-axis. If None, using "linear" for "time", "height", or "T" (singular)
        and "log" for "diam" (ABIFM).
    yscale: str or None
        "linear" or "log" scale for y-axis. If None, using "linear" for "time", "height", or "T" (singular)
        and "log" for "diam" (ABIFM).
    log_plot: bool
        scale for the c-axis (color-scale). Choose between linear (False) or log-scale (True).
    grid: bool
        If True, add grid.
    xlabel: str or None
        set xlabel with str. Use field name and units if None
    ylabel: str or None
        set ylabel with str. Use field name and units if None
    tight_layout: bool
        If True, using a tight layout (no text overlap or text cut at figure edges) - typically, the
        desired behavior).
    font_size: float or None
        set font size in panel
    xtick: list, np.ndarray, or None
        if provided, then used as x-axis ticks (check that values match the current dim units).
    xticklabel: list, np.ndarray, or None
        if provided, then used as x-axis tick labels.
    ytick: list, np.ndarray, or None
        if provided, then used as y-axis ticks (check that values match the current dim units).
    yticklabel: list, np.ndarray, or None
        if provided, then used as y-axis tick labels.
    xlim: 2-elemnt tuple (or list) or None
        xlim range (check that values match the current dim units).
    ylim: 2-elemnt tuple (or list) or None
        ylim range (check that values match the current dim units).
    """
    if title is not None:
        ax.set_title(title)

    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)

    if grid:
        ax.grid()

    if xscale is not None:
        ax.set_xscale(xscale)
    if yscale is not None:
        ax.set_yscale(yscale)

    if font_size is not None:
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                     ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(font_size)

    if xtick is not None:
        ax.set_xticks(xtick)
    if ytick is not None:
        ax.set_yticks(ytick)

    if xticklabel is not None:
        ax.set_xticklabels(xticklabel)
    if yticklabel is not None:
        ax.set_yticklabels(yticklabel)

    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    if tight_layout:
        plt.tight_layout()

    return ax


def process_dim(plot_data, dim_name, dim_vals_inds, dim_treat="sum"):
    """
    Process non-depicted dimension by cropping, slicing, averaging, summing, or calculating SD.

    Parameters
    ----------
    plot_data: xr.DataArray
        data array to be plotted.
    dim_name: str
        name of array coordinates to treat.
    dim_vals_inds: float, int, 2-element tuple, list, or None
        elements, range, or values to treat from the dim. Options:
        1. float to specify a dim coordinate value to use for plotting (check that values match current dim units).
        2. int for coordinate index.
        3. tuple of floats to define a range of values (check that values match the current dim units).
        4. tuple of ints to define a range of indices.
        5. list or np.ndarray of floats to define specific values (check that values match the current dim units).
        6. list or np.ndarray of ints to define specific indices.
        7. None to take the full coordinate range.
    dim_treat: str
        Relevant if dim_vals_inds is a tuple, list, np.ndarray, or None.
        Use "mean", "sum", or "sd" for mean, sum, or standard deviation, respectively.

    Returns
    -------
    plot_data: xr.DataArray
        data array to be plotted.

    """
    dim_ind = np.argwhere([dim_name == x for x in plot_data.dims]).item()
    if dim_treat is None:
        treat_fun = lambda x: x  # Do nothing.
    elif dim_treat == "sum":
        treat_fun = lambda x: np.sum(x, axis=dim_ind)
    elif dim_treat == "mean":
        treat_fun = lambda x: np.mean(x, axis=dim_ind)
    elif dim_treat == "sd":
        treat_fun = lambda x: np.std(x, axis=dim_ind)
    else:
        raise RuntimeError("'dim_treat' should be one of 'sum', 'mean', 'sd', or None")

    units = ""
    if "units" in plot_data.attrs:
        units = plot_data.attrs["units"]
    if dim_vals_inds is None:
        plot_data = treat_fun(plot_data)
    elif isinstance(dim_vals_inds, float):
        plot_data = plot_data.sel({dim_name: dim_vals_inds}, method="nearest")
    elif isinstance(dim_vals_inds, int):
        plot_data = plot_data.isel({dim_name: dim_vals_inds})
    elif isinstance(dim_vals_inds, tuple):
        if len(dim_vals_inds) != 2:
            raise RuntimeError("tuple (range) length must be 2")
        if isinstance(dim_vals_inds[0], float):  # check type of first index
            plot_data = treat_fun(plot_data.sel({dim_name: slice(dim_vals_inds[0], dim_vals_inds[1])}))
        else:
            plot_data = treat_fun(plot_data.isel({dim_name: slice(dim_vals_inds[0], dim_vals_inds[1])}))
    elif isinstance(dim_vals_inds, (list, np.ndarray)):
        if isinstance(dim_vals_inds[0], float):  # check type of first index
            plot_data = treat_fun(plot_data.sel({dim_name: dim_vals_inds}, method="nearest"))
        else:
            plot_data = treat_fun(plot_data.isel({dim_name: dim_vals_inds}))

    # restore units
    if units != "":
        plot_data.attrs["units"] = units

    return plot_data


def set_out_field_namelists(ci_model, plot_type):
    """
    set namelists for model output fields in preparation of plotting routines.

    Parameters
    ----------
    ci_model: ci_model class
        Containing model run output.
    plot_type: str
        Choose between "curtain", "profile", and "tseries", "psd".

    Returns
    -------
    aer_pop_aux_fields: list
        names of optional fields per aerosol population.
    aer_pop_w_diams: list
        names of optional fields per aerosol population with the diameter dimension.
    aer_pop_w_diams_str: dict
        per key, long str of field name
    diverging_vars: list
        names of optional fields per aerosol population that need to be plotted using
        diverging colormaps.
    """
    if plot_type == "curtain":
        if ci_model.prognostic_inp:
            aer_pop_aux_fields = ["Jhet", "ns_raw", "inp_pct"]
            aer_pop_w_diams = ["inp_tot", "n_aer", "budget_aer_mix", "budget_aer_act"]
            aer_pop_w_diams_str = {"inp": "INP T spec.", "inp_tot": "INP conc.", "n_aer": "conc.",
                                   "budget_aer_mix": "mixing budget", "budget_aer_act": "activation budget"}
            diverging_vars = ["budget_aer_mix", "budget_aer_act"]
        else:
            aer_pop_aux_fields = ["Jhet"]
            aer_pop_w_diams = ["n_aer"]
            if ci_model.use_ABIFM:
                aer_pop_w_diams += ["budget_aer_mix", "budget_aer_act"]
                diverging_vars = ["budget_aer_mix", "budget_aer_act"]
            else:
                diverging_vars = []
            aer_pop_w_diams_str = {"n_aer": "conc.",
                                   "budget_aer_mix": "mixing budget", "budget_aer_act": "activation budget"}
        return aer_pop_aux_fields, aer_pop_w_diams, aer_pop_w_diams_str, diverging_vars
    elif plot_type == "tseries":
        if ci_model.prognostic_inp:
            aer_pop_aux_fields = ["Jhet", "ns_raw", "inp_pct", "budget_aer_act", "pbl_aer_tot_rel_frac",
                                  "pbl_aer_tot_decay_rate", "pbl_inp_tot_rel_frac", "pbl_inp_mean"]
            aer_pop_w_diams = ["inp_tot", "n_aer", "budget_aer_mix", "budget_aer_ent"]
            aer_pop_w_diams_str = {"inp_snap": "INP T spec.", "inp_tot": "INP conc.", "n_aer": "conc.",
                                   "budget_aer_mix": "mix budget", "budget_aer_ent": "entrainment budget",
                                   "budget_aer_act": "activation budget"}
        else:
            aer_pop_w_diams = ["n_aer"]
            if ci_model.use_ABIFM:
                aer_pop_aux_fields = ["Jhet", "budget_aer_act"]
                aer_pop_w_diams += ["budget_aer_mix", "budget_aer_ent"]
            else:
                aer_pop_aux_fields = ["Jhet", "budget_aer_act", "pbl_inp_mean"]
            aer_pop_w_diams_str = {"n_aer": "conc.",
                                   "budget_aer_mix": "mix budget", "budget_aer_ent": "entrainment budget",
                                   "budget_aer_act": "activation budget"}
        return aer_pop_aux_fields, aer_pop_w_diams, aer_pop_w_diams_str
    elif plot_type == "profile":
        if ci_model.prognostic_inp:
            aer_pop_aux_fields = ["Jhet", "ns_raw", "inp_pct", "pbl_aer_tot_rel_frac", "pbl_aer_tot_decay_rate",
                                  "pbl_inp_tot_rel_frac", "pbl_inp_mean"]
            aer_pop_w_diams = ["inp_tot", "n_aer", "budget_aer_mix"]
            aer_pop_w_diams_str = {"inp": "INP T spec.", "inp_tot": "INP conc.", "n_aer": "conc.",
                                   "budget_aer_mix": "mix budget", "budget_aer_act": "activation budget"}
        else:
            aer_pop_aux_fields = ["Jhet"]
            aer_pop_w_diams = ["n_aer"]
            if ci_model.use_ABIFM:
                aer_pop_w_diams += ["budget_aer_mix"]
            aer_pop_w_diams_str = {"n_aer": "conc.",
                                   "budget_aer_mix": "mix budget", "budget_aer_act": "activation budget"}
        return aer_pop_aux_fields, aer_pop_w_diams, aer_pop_w_diams_str
    elif plot_type == "psd":
        if ci_model.prognostic_inp:
            aer_pop_w_diams = ["inp_tot", "n_aer"]
            aer_pop_w_diams_str = {"inp": "INP T spec.", "inp_tot": "INP conc.", "n_aer": "conc."}
        else:
            aer_pop_w_diams = ["n_aer"]
            aer_pop_w_diams_str = {"n_aer": "conc."}
        return aer_pop_w_diams, aer_pop_w_diams_str
