from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import matplotlib.pyplot as plt
import numpy as np

colors = [(0, 0, 0), (1, 0, 0)]  # cmap goes from black (0, 0, 0) to red (1, 0, 0)
fluo_cmap_red = LinearSegmentedColormap.from_list("Custom", colors, N=256)

colors = [(0, 0, 0), (0, 1, 0)]  # cmap goes from black (0, 0, 0) to green (1, 0, 0)
fluo_cmap_green = LinearSegmentedColormap.from_list("Custom", colors, N=256)

rng = np.random.default_rng(0)
label_cmap = ListedColormap(rng.random(size=(50, 3)))
label_cmap.set_bad([0, 0, 0])


def show_segmentation(
    dat, masks, clims=None, fluo_cmap=fluo_cmap_red, label_cmap=label_cmap
):
    # max_proj = signal_data_reg.max(axis=0)
    if isinstance(fluo_cmap, str):
        if fluo_cmap[0].lower() == "r":
            fluo_cmap = fluo_cmap_red
        elif fluo_cmap[0].lower() == "g":
            fluo_cmap = fluo_cmap_green

    if clims is None:
        clims = np.quantile(dat, [0.025, 0.995])
    fig, ax = plt.subplots(1, 2, figsize=(6, 3), sharex=True, sharey=True)
    masks_plt = masks.copy().astype("float")
    masks_plt[masks_plt == 0] = np.nan
    h = ax[0].imshow(dat, cmap=fluo_cmap, vmin=clims[0], vmax=clims[1])
    ax[1].imshow(masks_plt, cmap=label_cmap)
    for _ax in ax:
        _ax.axis("off")
    ax[0].set_title("Max proj")
    ax[1].set_title("ROI masks")
    # fig.suptitle(fname)
    fig.tight_layout()
    cbar = fig.colorbar(h, ax=ax)
    cbar.set_label("Intensity (AU)")
    return fig, ax


# def plot_bleaching_trace(dat):
#     import seaborn as sns
#     fig, ax = plt.subplots(1, figsize=(4, 2))
#     sns.lineplot(dat, x="t", y="value", ax=ax)
#     ax.set_ylabel("Pixel intensity")
#     ax.set_xlabel("Time (frames)")
#     return fig, ax

def plot_trace(dat, x="t", y="value", phases=None, ylabel=None, exclude_first_points=False):
    # mark the beginning of each phase with a vertical line...
    # EXCLUDE first point of phase as an option
    import seaborn as sns
    fig, ax = plt.subplots(1, figsize=(4, 2))
    plt_dat = dat.copy()
    
    first_points = []
    if phases is not None:
        for frames in phases.values():
            frame_lst = list(frames)
            try:
                first_points.append(plt_dat.loc[plt_dat["frame_number"]==min(frame_lst), "t"].iloc[0])
            except IndexError:
                pass
            # t = dat.loc[dat["frame_number"]==frame_lst[0], "t"].iloc[0]
            # ax.axvline(t, alpha=.5) # mark the beginning of each phase..
    
    if exclude_first_points:
        plt_dat.loc[plt_dat["t"].isin(first_points), y] = np.nan
    sns.lineplot(plt_dat, x=x, y=y, ax=ax)
    for _point in first_points:
        ax.axvline(_point, alpha=.3) 
    if ylabel is None:
        ax.set_ylabel("Pixel intensity")
    else:
        ax.set_ylabel(ylabel)
    ax.set_xlabel("Time (frames)")
    return fig, ax