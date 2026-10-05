import os
os.environ["CUDA_VISIBLE_DEVICES"] = "8"

from spyglass.decoding.v1.c3po import Model
from spyglass.common import Nwbfile

# sel_key = {'nwb_file_name': 'j1620210710_.nwb',
#  'marks_group_name': 'HPC',
#  'model_name': 'j16_22tet',
#  'training_params_name': 'wilbur_post_fix',
#  'training_interval_name': 'runs_noPrePostTrialTimes raw data valid times'}

# sel_key = {'nwb_file_name': 'j1620210710_.nwb',
#  'marks_group_name': 'HPC',
#  'model_name': 'j16_22tet',
#  'training_params_name': 'wilbur_post_fix',
#  'training_interval_name': 'pos 1 valid times'}

# sel_key = {'nwb_file_name': 'SC9220250228_.nwb',
#  'marks_group_name': 'mpfc sorted spikes',
#  'model_name': 'SC92_sorted_spikes_vFastDynamics',
#  'training_params_name': 'prelim_mpfc_wf',
#  'training_interval_name': 'run epoch valid times'}

# sel_key = {'nwb_file_name': 'RS5820251203_.nwb',
#  'training_params_name': 'RS58_sorted_v1',
#  'model_name': 'RS58_epoch06_ca1_v0',
#  'marks_group_name': '06_r3_sorted_spikes',
#  'training_interval_name': '06_r3'}

# sel_key = {'nwb_file_name': 'RS5820251203_.nwb',
#  'training_params_name': 'RS58_sorted_v1',
#  'model_name': 'RS58_epoch12_ca1_v0',
#  'marks_group_name': '12_r6_sorted_spikes',
#  'training_interval_name': '12_r6'}

# sel_key = {'nwb_file_name': 'RS5820251203_.nwb',
#  'training_params_name': 'RS58_clusterless_v1',
#  'model_name': 'RS58_clusterless_ca1_v0',
#  'marks_group_name': '06_r3_clusterless',
#  'training_interval_name': '06_r3'}

# sel_key = {'nwb_file_name': 'mango20211129_.nwb',
#  'training_params_name': 'wilbur_post_fix',
#  'model_name': 'mango20211129_sorted_spikes_v0',
#  'marks_group_name': 'mpfc_sorted_spikes',
#  'training_interval_name': '15_s8'}

# sel_key = {'nwb_file_name': 'mango20211129_.nwb',
#  'training_params_name': 'wilbur_post_fix',
#  'model_name': 'mango20211129_sorted_spikes_v1',
#  'marks_group_name': 'mpfc_sorted_spikes',
#  'training_interval_name': '15_s8'}

# sel_key = {'nwb_file_name': 'Wallie20220912_.nwb',
#  'training_params_name': 'wilbur_post_fix',
#  'model_name': 'wallie_linear_dim20_noSmooth_v0',
#  'marks_group_name': '10_lineartrack_waveforms',
#  'training_interval_name': '10_lineartrack'}

# sel_key = {'nwb_file_name': 'Wallie20220912_.nwb',
#  'training_params_name': 'wilbur_post_fix',
#  'model_name': 'wallie_linear_dim20_10Smooth_v0',
#  'marks_group_name': '10_lineartrack_waveforms',
#  'training_interval_name': '10_lineartrack'}

# sel_key = {'nwb_file_name': 'Wallie20220912_.nwb',
#  'training_params_name': 'wallie_20_dim_v0',
#  'model_name': 'wallie_linear_dim20_10Smooth_v0',
#  'marks_group_name': '10_lineartrack_waveforms',
#  'training_interval_name': '10_lineartrack'}

# sel_key = {'nwb_file_name': 'Wallie20220912_.nwb',
#  'training_params_name': 'wallie_20_dim_v0',
#  'model_name': 'wallie_linear_dim20_10Smooth_v0',
#  'marks_group_name': '10_lineartrack_wf_norm1000',
#  'training_interval_name': '10_lineartrack'}

# sel_key ={'nwb_file_name': 'Wallie20220912_.nwb',
#  'training_params_name': 'wallie_20_dim_v0',
#  'model_name': 'wallie_linear_dim20_noSmooth_v0',
#  'marks_group_name': '10_lineartrack_wf_norm1000',
#  'training_interval_name': '10_lineartrack'}

# sel_key = {'nwb_file_name': 'Wallie20220912_.nwb',
#  'training_params_name': 'wallie_20_dim_v0',
#  'model_name': 'wallie_dim20_HYBRID_10Smooth_v0',
#  'marks_group_name': '10_lineartrack_HYBRID',
#  'training_interval_name': '10_lineartrack'}

# sel_key = {'nwb_file_name': 'wilbur20210404_.nwb',
#  'training_params_name': 'wilbur_post_fix',
#  'model_name': 'wilbur_v0_HYBRID_20210404',
#  'marks_group_name': 'mpfc_HYBRID',
#  'training_interval_name': 'all_run_epochs'}

# sel_key = {'nwb_file_name': 'wilbur20210406_.nwb',
#  'training_params_name': 'wilbur_post_fix',
#  'model_name': 'wilbur_v0_HYBRID_20210406',
#  'marks_group_name': 'mpfc_HYBRID',
#  'training_interval_name': 'all_run_epochs'}

# sel_key = {'nwb_file_name': 'wilbur20210406_.nwb',
#  'training_params_name': 'wilbur_big_model',
#  'model_name': 'wilbur_v0_HYBRID_20210406_128d',
#  'marks_group_name': 'mpfc_HYBRID',
#  'training_interval_name': 'all_run_epochs'}

# sel_key = {'nwb_file_name': 'Wallie20220912_.nwb',
#  'training_params_name': 'wallie_20_dim_v0',
#  'model_name': 'wallie_linear_dim8_noSmooth_v0',
#  'marks_group_name': '10_lineartrack_wf_norm1000',
#  'training_interval_name': '10_lineartrack'}

# sel_key = {'nwb_file_name': 'Wallie20220912_.nwb',
#  'marks_group_name': '10_lineartrack_wf_norm1000',
#  'model_name': 'wallie_linear_dim20_noSmooth_v0',
#  'training_params_name': 'wallie_20_dim_v0',
#  'training_interval_name': '10_lineartrack'}

sel_key = {'nwb_file_name': 'RS5820251203_.nwb',
 'training_params_name': 'RS58_clusterless_v1',
 'model_name': 'RS58_clusterless_ca1_v0',
 'marks_group_name': '06_r3_clusterless',
 'training_interval_name': '06_r3'}

sel_key = {'nwb_file_name': 'RS5820251203_.nwb',
 'training_params_name': 'RS58_clusterless_v1',
 'model_name': 'RS58_clusterless_ca1_smooth_v0',
 'marks_group_name': '06_r3_clusterless',
 'training_interval_name': '06_r3'}

sel_key = {'nwb_file_name': 'RS5820251203_.nwb',
 'training_params_name': 'RS58_clusterless_v1',
 'model_name': 'RS58_clusterless_ca1_smoothn2_v0',
 'marks_group_name': '06_r3_clusterless',
 'training_interval_name': '06_r3'}

sel_key = {'nwb_file_name': 'RS5820251203_.nwb',
 'training_params_name': 'RS58_clusterless_v1',
 'model_name': 'RS58_clusterless_ca1_smooth2_d32',
 'marks_group_name': '06_r3_clusterless',
 'training_interval_name': '06_r3'}

sel_key = {'nwb_file_name': 'RS5820251203_.nwb',
 'training_params_name': 'RS58_clusterless_full_day',
 'model_name': 'RS58_clusterless_ca1_smoothn2_v0',
 'marks_group_name': 'all_run_epochs_clusterless',
 'training_interval_name': 'all_run_epochs'}

sel_key = {'nwb_file_name': 'Winnie20220714_.nwb',
 'training_params_name': 'wallie_20_dim_v0',
 'model_name': 'winnie_linear_dim20_noSmooth_v0',
 'marks_group_name': '10_lineartrack_wf_norm1000',
 'training_interval_name': '10_lineartrack'}


def main():
    (Nwbfile() & sel_key).fetch_nwb()
    Model().populate(sel_key)

if __name__ == "__main__":
    main()

