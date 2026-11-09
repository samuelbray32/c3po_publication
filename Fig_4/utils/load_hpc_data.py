# load position data
from spyglass.common import PositionIntervalMap
from spyglass.decoding.v1.core import PositionGroup
from spyglass.position.position_merge import PositionOutput
from spyglass.linearization.v1 import LinearizedPositionV1
from scipy.ndimage import gaussian_filter1d

from spyglass.lfp.analysis.v1 import LFPBandV1
from spyglass.lfp import LFPOutput

import numpy as np


def load_position_data(nwb_file_name, epoch_interval):

    epoch_key = {
        "nwb_file_name": nwb_file_name,
        "interval_list_name": epoch_interval,
    }
    pos_interval = (PositionIntervalMap() & epoch_key).fetch1("position_interval_name")
    pos_key = {
        "nwb_file_name": nwb_file_name,
        "interval_list_name": pos_interval,
        "trodes_pos_params_name": "single_led_upsampled",
    }
    pos_merge_id = (PositionOutput.TrodesPosV1() & pos_key).fetch("merge_id")
    pos_merge_key = {"pos_merge_id": pos_merge_id[0]}


    pos_df = (LinearizedPositionV1() & pos_merge_key).fetch1_dataframe()
    pos_times = pos_df.index.values  # - first_mark_time
    pos = pos_df.linear_position.values
    vel = np.diff(pos_df.linear_position.values)
    vel = np.concatenate([vel, [vel[-1]]])
    vel = gaussian_filter1d(vel, 11, axis=0, mode="nearest")

    # Define position-based intervals
    all_running = (np.abs(vel * 500) > 10).astype(int)
    right_running = ((vel * 500) > 10).astype(int)
    left_running = ((vel * 500) < -10).astype(int)

    st = np.where(np.diff(all_running) == 1)[0]
    en = np.where(np.diff(all_running) == -1)[0]
    if en[0] < st[0]:
        en = en[1:]
    all_running_intervals = np.array(
        [[pos_df.index[s], pos_df.index[e]] for s, e in zip(st, en)]
    )
    st = np.where(np.diff(left_running) == 1)[0]
    en = np.where(np.diff(left_running) == -1)[0]
    if en[0] < st[0]:
        en = en[1:]
    left_running_intervals = np.array(
        [[pos_df.index[s], pos_df.index[e]] for s, e in zip(st, en)]
    )

    st = np.where(np.diff(right_running) == 1)[0]
    en = np.where(np.diff(right_running) == -1)[0]
    if en[0] < st[0]:
        en = en[1:]
    right_running_intervals = np.array(
        [[pos_df.index[s], pos_df.index[e]] for s, e in zip(st, en)]
    )

    return {
        "pos_df": pos_df,
        "pos_times": pos_times,
        "pos": pos,
        "vel": vel,
        "all_running_intervals": all_running_intervals,
        "left_running_intervals": left_running_intervals,
        "right_running_intervals": right_running_intervals
    }

def load_lfp_data(nwb_file_name, epoch_interval):
    ref_ind = 12  # hardcoded for dataset Wallie20220912_.nwb
    if not nwb_file_name == "Wallie20220912_.nwb":
        raise ValueError("This function is currently hardcoded for Wallie20220912_.nwb")

    epoch_key = {
        "nwb_file_name": nwb_file_name,
        "interval_list_name": epoch_interval,
    }
    pos_interval = (PositionIntervalMap() & epoch_key).fetch1("position_interval_name")
    pos_key = {
        "nwb_file_name": nwb_file_name,
        "interval_list_name": pos_interval,
        "trodes_pos_params_name": "single_led_upsampled",
    }


    lfp_key = {
        "nwb_file_name": pos_key["nwb_file_name"],
        "target_interval_list_name": pos_key["interval_list_name"],
    }

    exclude = (
        (LFPOutput.LFPV1() & {"lfp_electrode_group_name": "tetrode_samples"})
        .proj(lfp_merge_id="merge_id")
        .fetch("lfp_merge_id", as_dict=True)
    )


    # Theta band
    theta_key = {
        "nwb_file_name": pos_key["nwb_file_name"],
        "target_interval_list_name": pos_key["interval_list_name"],
        "filter_name": "Theta 5-11 Hz",
    }
    query = (LFPBandV1() & theta_key) - exclude

    theta_band_obj = query.fetch_nwb()[0]["lfp_band"]
    phase_df = query.compute_signal_phase([ref_ind])
    power_df = query.compute_signal_power([ref_ind])
    phase_df["phase"] = phase_df.values
    phase_df["power"] = power_df.values
    data_column = [name for name in phase_df.columns if "electrode" in name][0]
    phase_df["signal"] = theta_band_obj.data[:, ref_ind]  # phase_df[data_column]

    # Fast Gamma band
    fast_gamma_key = {
        "nwb_file_name": pos_key["nwb_file_name"],
        "target_interval_list_name": pos_key["interval_list_name"],
        "filter_name": "Fast Gamma 65-100 Hz",
        # "filter_name": "Slow Gamma 25-55 Hz",
    }
    query = (LFPBandV1() & fast_gamma_key) - exclude
    band_obj = query.fetch_nwb()[0]["lfp_band"]
    fg_phase_df = query.compute_signal_phase([ref_ind])
    fg_power_df = query.compute_signal_power([ref_ind])
    fg_phase_df["phase"] = fg_phase_df.values
    fg_phase_df["power"] = fg_power_df.values
    data_column = [name for name in fg_phase_df.columns if "electrode" in name][0]
    fg_phase_df["signal"] = band_obj.data[:, ref_ind]  # fg_phase_df[data_column]

    # Slow Gamma band
    slow_gamma_key = {
        "nwb_file_name": pos_key["nwb_file_name"],
        "target_interval_list_name": pos_key["interval_list_name"],
        "filter_name": "Slow Gamma 25-55 Hz",
    }
    query = (LFPBandV1() & slow_gamma_key) - exclude
    band_obj = query.fetch_nwb()[0]["lfp_band"]
    sg_phase_df = query.compute_signal_phase([ref_ind])
    sg_power_df = query.compute_signal_power([ref_ind])
    sg_phase_df["phase"] = sg_phase_df.values
    sg_phase_df["power"] = sg_power_df.values
    data_column = [name for name in sg_phase_df.columns if "electrode" in name][0]
    sg_phase_df["signal"] = band_obj.data[:, ref_ind]  # sg_phase_df[data_column]

    # # Ripple band
    # ripple_band_key = {
    #     "nwb_file_name": pos_key["nwb_file_name"],
    #     # "target_interval_list_name": pos_key["interval_list_name"],
    #     "filter_name": "Ripple 150-250 Hz",
    # }

    # query = (
    #     LFPBandV1
    #     & ripple_band_key
    #     & f"target_interval_list_name LIKE '%{pos_key['interval_list_name']}%'"
    # )
    # ripple_band_obj = query.fetch_nwb()[0]["lfp_band"]

    return {
        "theta_phase_df": phase_df,
        "fg_phase_df": fg_phase_df,
        "sg_phase_df": sg_phase_df,
    }