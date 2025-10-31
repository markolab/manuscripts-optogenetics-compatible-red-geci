import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from skimage.measure import regionprops_table
from calcium_imaging_analysis.io import short_name, write_video
from calcium_imaging_analysis.viz import show_segmentation, plot_trace


def proc_photoswitch(
    proc_fname,
    output_fig_dir=None,
    data_channel=["mCherry"],
    dff0_quantile=0.1,
    force=False,
):

    save_fname = os.path.splitext(proc_fname)[0] + ".parquet"
    if os.path.exists(save_fname) and (not force):
        traces = pd.read_parquet(save_fname)
        return traces

    data_dct = joblib.load(proc_fname)

    if "timesteps" not in data_dct.keys():
        return None

    timesteps = data_dct["timesteps"]
    all_data_reg = data_dct["registered_frames"]
    phases = data_dct["phases"]
    phases_first_timestep = data_dct["phases_first_timestep"]
    masks = data_dct["roi_masks"]
    channels = data_dct["channels"]
    well_name = data_dct["well_name"]
    aux_image = data_dct["registered_aux_image"]

    proc_dir = os.path.dirname(os.path.normpath(proc_fname))
    session_name = proc_dir.split(os.path.sep)[-2]
    fname = os.path.splitext(os.path.basename(proc_fname))[0]
    short_fname = short_name(fname)

    # check for aux image and regionprops mean_intensity

    if output_fig_dir is not None:
        os.makedirs(os.path.join(output_fig_dir, session_name), exist_ok=True)
        sanitized_filename = os.path.join(
            output_fig_dir, session_name, f"{fname}-session-{session_name}"
        )

    try:
        experiment_type = data_dct["experiment_type"]
    except KeyError:
        experiment_type = "timecourse"

    signal_data_reg = None
    for _channel in data_channel:
        try:
            signal_data_reg = all_data_reg.take(channels.index(_channel), axis=0)
            break
        except ValueError:
            pass

    stats = [
        regionprops_table(masks, intensity_image=_im, properties=["mean_intensity"])[
            "mean_intensity"
        ]
        for _im in signal_data_reg
    ]

    df = pd.DataFrame(np.array(stats))
    df.index.name = "timestep"
    traces = df
    traces_name = "Raw Fluo."
    traces_ratio_name = "Raw Ratio"

    # traces_df0 = traces.rolling(10, 1, True).quantile(.1)
    # traces_df0 = traces.iloc[:20].median()
    # traces = (traces - traces_df0) / traces_df0
    # traces *= 1e2
    traces["t"] = timesteps
    traces["frame_number"] = np.arange(len(traces))
    traces["phase"] = "n/a"
    for k, v in phases.items():
        traces.loc[traces.index.isin(v), "phase"] = k
    if len(phases) > 0:
        traces = traces.loc[traces["phase"] != "n/a"]
        traces["t_align"] = traces.groupby("phase")["t"].transform(
            lambda x: x - phases_first_timestep[x.name]
        )
        # traces = traces.melt(var_name="roi", id_vars=["t","phase","t_align"], ignore_index=False).reset_index()
    else:
        traces["t_align"] = traces["t"] - traces["t"].iat[0]

    traces = traces.melt(
        id_vars=["phase", "t", "t_align", "frame_number"], var_name="roi"
    )
    traces["well"] = well_name
    traces["well_sanitized"] = traces["well"].apply(short_name)
    traces["filename"] = proc_fname
    traces["value_dff0"] = traces.groupby(["well", "roi"])["value"].transform(
        lambda x: (x - x.quantile(dff0_quantile)) / x.quantile(dff0_quantile)
    )

    if aux_image is not None:
        stats_aux = regionprops_table(masks, intensity_image=aux_image, properties=["mean_intensity"])
        stats_aux = stats_aux["mean_intensity"]
        stats_aux = {i: val for i, val in enumerate(stats_aux)}
        traces["value_aux"] = traces["roi"].map(stats_aux)
    else:
        stats_aux = {}

    traces.to_parquet(save_fname)

    if output_fig_dir is not None:
        clims = np.quantile(signal_data_reg, [0.025, 0.995])
        fig, ax = show_segmentation(np.max(signal_data_reg, axis=0), masks, clims=clims, fluo_cmap="r")
        fig.suptitle(fname)
        fig.savefig(f"{sanitized_filename}-max_proj.png", dpi=300, bbox_inches="tight")
        
        if aux_image is not None:
            aux_clims = np.quantile(aux_image, [0.025, 0.995])
            fig, ax = show_segmentation(aux_image, masks, clims=aux_clims, fluo_cmap="g")
            fig.suptitle(fname)
            fig.savefig(f"{sanitized_filename}-aux_image.png", dpi=300, bbox_inches="tight")
        
        fig, ax = plot_trace(traces, phases=phases, x="t", exclude_first_points=False)
        fig.suptitle(fname)
        fig.savefig(
            f"{sanitized_filename}-timecourse-raw.png", dpi=300, bbox_inches="tight"
        )
        fig, ax = plot_trace(traces, y="value_dff0", ylabel="dF/F0", phases=phases, x="t", exclude_first_points=False)
        fig.suptitle(fname)
        fig.savefig(
            f"{sanitized_filename}-timecourse-dff0.png", dpi=300, bbox_inches="tight"
        )

        # for now pin to 10, but let's change to get accurate timestamps from file directly...
        write_video(
            signal_data_reg,
            f"{sanitized_filename}-clims-{clims}.mp4",
            phases=phases,
            movie_fps=10,
            clims=clims,
            threads=1,
        )

    return traces


def proc_photobleach(
    proc_fname, output_fig_dir=None, force=False, data_channel=["TRITC"]
):

    save_fname = os.path.splitext(proc_fname)[0] + ".parquet"
    if os.path.exists(save_fname) and (not force):
        traces = pd.read_parquet(save_fname)
        return traces

    proc_dir = os.path.dirname(os.path.normpath(proc_fname))
    session_name = proc_dir.split(os.path.sep)[-2]
    fname = os.path.splitext(os.path.basename(proc_fname))[0]
    short_fname = short_name(fname)

    if output_fig_dir is not None:
        os.makedirs(os.path.join(output_fig_dir, session_name), exist_ok=True)
        sanitized_filename = os.path.join(
            output_fig_dir, session_name, f"{short_fname}-session-{session_name}"
        )

    data_dct = joblib.load(proc_fname)
    timesteps = data_dct["timesteps"]
    all_data_reg = data_dct["registered_frames"]
    phases = data_dct["phases"]
    phases_first_timestep = data_dct["phases_first_timestep"]
    masks = data_dct["roi_masks"]
    channels = data_dct["channels"]
    well_name = data_dct["well_name"]

    try:
        experiment_type = data_dct["experiment_type"]
    except KeyError:
        experiment_type = "timecourse"

    signal_data_reg = None
    for _channel in data_channel:
        try:
            signal_data_reg = all_data_reg.take(channels.index(_channel), axis=0)
            break
        except ValueError:
            pass

    stats = [
        regionprops_table(masks, intensity_image=_im, properties=["mean_intensity"])[
            "mean_intensity"
        ]
        for _im in signal_data_reg
    ]

    df = pd.DataFrame(np.array(stats))
    df.index.name = "timestep"
    traces = df
    traces_name = "Raw Fluo."
    traces_ratio_name = "Raw Ratio"

    # traces_df0 = traces.rolling(10, 1, True).quantile(.1)
    # traces = (traces - traces_df0) / traces_df0
    # only normalize post-hoc...
    # traces = traces / traces.iloc[0]
    traces["t"] = timesteps - timesteps[0]
    traces["frame_number"] = np.arange(len(traces))
    traces["phase"] = "n/a"
    for k, v in phases.items():
        traces.loc[traces.index.isin(v), "phase"] = k
    if len(phases) > 0:
        traces = traces.loc[traces["phase"] != "n/a"]
        traces["t_align"] = traces.groupby("phase")["t"].transform(
            lambda x: x - phases_first_timestep[x.name]
        )
        # traces = traces.melt(var_name="roi", id_vars=["t","phase","t_align"], ignore_index=False).reset_index()
    else:
        traces["t_align"] = traces["t"] - traces["t"].iat[0]

    traces = traces.melt(
        id_vars=["phase", "t", "t_align", "frame_number"], var_name="roi"
    )
    traces["well"] = well_name
    traces["well_sanitized"] = traces["well"].apply(short_name)
    traces["filename"] = proc_fname
    traces.to_parquet(save_fname)

    if output_fig_dir is not None:
        clims = np.quantile(signal_data_reg, [0.025, 0.995])
        fig, ax = show_segmentation(np.max(signal_data_reg, axis=0), masks, clims=clims)
        fig.suptitle(fname)
        fig.savefig(f"{sanitized_filename}-max_proj.png", dpi=300, bbox_inches="tight")
        fig, ax = plot_trace(traces)
        fig.suptitle(fname)
        fig.savefig(
            f"{sanitized_filename}-timecourse.png", dpi=300, bbox_inches="tight"
        )
        write_video(signal_data_reg, f"{sanitized_filename}.mp4", clims=clims)

    return traces
