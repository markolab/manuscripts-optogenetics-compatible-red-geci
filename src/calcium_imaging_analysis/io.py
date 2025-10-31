import re
import numpy as np
from markovids import vid
from calcium_imaging_analysis.viz import fluo_cmap_red

strip_list = [
    r"[a-z|A-Z]\.[0-9]+\_",  # well number
    r"[p|P]late[0-9]*\_",  # plate number
    r"JM[0-9]+\_",
    r"NES\_",
    r"photobleachv3\_",
    r"tet\-on\_",
]
phase_titles = ["baseline", "ionomycin", "egta"]


def short_name(string):
    tmp = library_name(string)
    if tmp is None:
        tmp = nonlibrary_name(string)
    return tmp


def library_name(string):
    tmp = re.match(r".*(s3[i]*[\_|-]l[0-9]+[\_|-][0-9]+)", string.lower())
    if tmp is None:
        return None
    else:
        match = tmp.group(1)
        tokens = re.match(r"^(s3[i]*)[\_|-]l([0-9]+)[\_|-]([0-9]+)$", match)
        # plate_number = int(tokens.group(0))
        base_sensor = tokens.group(1)
        library_number = int(tokens.group(2))
        variant_number = int(tokens.group(3))
        sanitized_name = f"{base_sensor}-l{library_number:01d}-{variant_number:03d}"
        return sanitized_name


def nonlibrary_name(string):
    use_string = string.lower()
    if "jrcamp1b" in use_string:
        return "jrcamp1b"
    elif "jrgeco1a" in use_string:
        return "jrgeco1a"
    elif any(_ in use_string for _ in ["no_plasmid", "negative"]):
        return "no_plasmid"
    elif "mscarlet3" in use_string:
        return "mscarlet3"
    elif "jrcamp1a" in use_string:
        return "jrcamp1a"
    else:
        tmp = use_string
        for _strip in strip_list:
            tmp = re.sub(_strip, "", tmp)
        return tmp.split("_")[0]


def write_video(dat, fname, phases=None, cmap=fluo_cmap_red, clims=None, movie_fps=2, **kwargs):

    if clims is None:
        clims = np.quantile(dat, [0.025, 0.995])

    # write out animation of data...
    writer_raw = vid.io.MP4WriterPreview(
        fname,
        frame_size=(dat.shape[2], dat.shape[1]),
        fps=movie_fps,
        cmap=cmap,
        **kwargs
    )
    writer_raw.open()
    writer_raw.write_frames(
        dat,
        vmin=clims[0],
        vmax=clims[1],
        mark_frames=(
            [] if phases is None else [_phase.start for _phase in phases.values()]
        ),
        progress_bar=False,
    )
    writer_raw.close()


def nd2_metadata_parse(img):
    import nd2

    # get
    wells = []
    phase_lens = []
    with nd2.ND2File(img) as f:
        for _experiment in f.experiment:
            if _experiment.type == "NETimeLoop":
                phase_lens = [_.count for _ in _experiment.parameters.periods]
            elif _experiment.type == "XYPosLoop":
                wells = [_.name for _ in _experiment.parameters.points]
                well_scan = True
        channel_names = [_.channel.name for _ in f.metadata.channels]
        events = f.events()
        # arr = f.asarray()
        dim_sizes = f.sizes

    if len(wells) > 0:
        timesteps = {}
        for _well in wells:
            timesteps[_well] = np.array(
                [
                    _["Time [s]"]
                    for _ in events
                    if ("Position Name" in _.keys()) and (_["Position Name"] == _well)
                ]
            )
    else:
        timesteps = np.array([_["Time [s]"] for _ in events if "T Index" in _.keys()])

    return_dct = {
        "dim_sizes": dim_sizes,
        "timesteps": timesteps,
        "channel_names": channel_names,
        "phase_lens": phase_lens,
        "events": events,
        "wells": wells,
        # "frames": arr
    }
    return return_dct
