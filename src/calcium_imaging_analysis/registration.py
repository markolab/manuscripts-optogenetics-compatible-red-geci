import numpy as np
import nd2
import os
import joblib
from calcium_imaging_analysis.viz import fluo_cmap_red, label_cmap
from calcium_imaging_analysis.io import nd2_metadata_parse
from pystackreg import StackReg
from cellpose import models

phase_titles_timecourse = ["baseline", "ionomycin", "egta"]
model_eval_kwargs = {"diameter": 80, "cellprob_threshold": 0.9, "channels": [[0, 0]]}

def register_and_get_rois(
    img,
    cellpose_model="cyto2",
    transform="rigid_body",
    model_eval_kwargs=model_eval_kwargs,
    well_name=None,  # only used for multi-well files
    roi_channel=["FITC", "TRITC", "mCherry"],  # use channels in this order for ROI
    experiment_type="timecourse",
):
    if transform.lower() == "affine":
        tf = StackReg.AFFINE
    elif transform.lower() == "rigid_body":
        tf = StackReg.RIGID_BODY
    else:
        raise RuntimeError("Transform not recognized")

    metadata = nd2_metadata_parse(img)
    with nd2.ND2File(img) as f:
        arr = f.asarray()
    dim_names = list((metadata["dim_sizes"].keys()))
    phase_lens = metadata["phase_lens"]
    if well_name is not None:
        timesteps = metadata["timesteps"][well_name]
    else:
        timesteps = metadata["timesteps"]
        
    if well_name is None:
        well_name = os.path.splitext(os.path.basename(img))[0]

    output_dir = os.path.join(os.path.dirname(img), "_analysis")
    sanitized_filename = os.path.join(output_dir, well_name)
    output_fname = f"{sanitized_filename}.p"

    if os.path.exists(output_fname):
        print(f"Path exists: {output_fname}")
        return None
    else:
        os.makedirs(output_dir, exist_ok=True)

    use_roi_channel = None
    for _channel in roi_channel:
        try:
            use_roi_channel = metadata["channel_names"].index(_channel)
            break
        except ValueError:
            pass
        
    if use_roi_channel is None:
        print(f"Unable to process {img}: no channel names")
        joblib.dump(
            {},
            output_fname,
            compress="lz4",
        )
        return None

    well_scan = len(metadata["wells"]) > 0

    # need to check for gaps in timesteps for manual recordings, other methods too unreliable
    if (not well_scan) and (experiment_type == "timecourse"):
        # look for 1 std above median
        df_timesteps = np.diff(timesteps)
        threshold = 1 * np.std(df_timesteps) + np.median(df_timesteps)
        phase_changes = list(np.flatnonzero(df_timesteps > threshold) + 1)
        phase_lens = np.diff(
            np.unique(np.array([0] + phase_changes + [len(timesteps)]))
        )

    edge = 0
    steps = []
    for _phase_len in phase_lens:
        steps.append(range(edge, edge + _phase_len))
        edge = steps[-1].stop

    if experiment_type == "timecourse":
        phases = {
            _phase: _steps for _phase, _steps in zip(phase_titles_timecourse, steps)
        }
    elif experiment_type == "photoswitch":
        phases = {f"pulse{i:02d}": _steps for i, _steps in enumerate(steps)}
    else:
        phases = {f"phase{i:02d}": _steps for i, _steps in enumerate(steps)}
    nchannels = len(metadata["channel_names"])

    # if we find multiple positions in the file that means we're doing a well scan
    phases_first_timestep = {}
    for k, v in phases.items():
        if well_scan and ("P Index" in metadata["events"][0].keys()):
            for _event in metadata["events"]:
                if (
                    ("P Index" in _event.keys())
                    and ("T Index" in _event.keys())
                    and (_event["P Index"] == 0)
                    and (_event["T Index"] == v.start)
                ):
                    phases_first_timestep[k] = _event["Time [s]"]
        else:
            phases_first_timestep[k] = timesteps[v.start]

    # if we can't find position, assume it's a single well
    try:
        use_well = metadata["wells"].index(well_name)
        arr = arr.take(use_well, axis=dim_names.index("P"))
        del [dim_names[dim_names.index("P")]]
    except ValueError:
        pass
    
    # if we can't find channels, assume it's single channel data
    try:
        arr_well_roi_channel = arr.take(use_roi_channel, axis=dim_names.index("C"))
    except ValueError:
        arr_well_roi_channel = arr

    if arr_well_roi_channel.ndim != 3:
        print(f"Unable to process {img}: not 3 dimensional data")
        joblib.dump(
            {},
            output_fname,
            compress="lz4",
        )
        return None
        
    try:
        nframes = len(arr_well_roi_channel)
    except TypeError as e:
        print(f"Unable to process {img}: bad array dimensions")
        print(e)
        joblib.dump(
            {},
            output_fname,
            compress="lz4",
        )
        return None

    model = models.Cellpose(model_type=cellpose_model)
    sr = StackReg(tf)
    tmats = sr.register_stack(
        arr_well_roi_channel,
        axis=0,
        reference="previous",
        verbose=False,
    )

    # transform all channels...
    # register_stack stores transformation matrices in self.tmats
    # these are then used to register the data...
    data_reg = []
    if "C" in dim_names:
        for i, _channel in enumerate(metadata["channel_names"]):
            data_reg.append(sr.transform_stack(arr.take(i, axis=dim_names.index("C"))))
    else:
        data_reg.append(sr.transform_stack(arr))
    data_reg = np.array(data_reg)

    max_proj = data_reg[use_roi_channel].max(axis=0)
    masks, _, _, _ = model.eval(max_proj, **model_eval_kwargs)

    joblib.dump(
        {
            "registered_frames": data_reg,
            "roi_masks": masks,
            "phases": phases,
            "phases_first_timestep": phases_first_timestep,
            "experiment_type": experiment_type,
            "timesteps": timesteps,
            "well_name": well_name,
            "well_scan": well_scan,
            "channels": metadata["channel_names"],
            "events": metadata["events"],
        },
        output_fname,
        compress="lz4",
    )