# c3po_publication

Notebooks and scripts associated with the upcoming publication: "Learning predictive latent structure from unclustered spike trains"

## Try it out

Try C3PO on a public DANDI recording in Google Colab with no local installation required. Enter the Dandiset ID and the path to an NWB file containing sorted spikes. The notebook sets up C3PO in the Colab runtime, trains a model on a GPU, and displays a PCA plot of the inferred context. You can download the trained model and analysis outputs as a ZIP archive or save them to Google Drive. [Open the C3PO demo in Google Colab](https://colab.research.google.com/github/samuelbray32/c3po/blob/main/tutorial/c3po_dandi_colab.ipynb).

## Replicate Locally

### **No Spyglass**

All figures besides 4-6 can be run locally by creating the following environment. (Assuming
conda install)

```bash
conda env create c3po
conda activate c3po
conda install pip
pip install c3po-neuro[cloud] # run pip install c3po-neuro[gpu,cloud] to use gpu acceleration
```

### **Spyglass dependent figures**

Figures 4-6 rely on a spyglass SQL database.  To run the code for these (and all other figures)
we provide a dockerized paper capable of accessing all data and software necessary for
complete paper replication. To run locally:

```bash
Todo: Instructions
```

## Code Directory

Panel letters follow the manuscript and Supplementary Information. Panels that share a notebook are grouped where practical. The table distinguishes figure schematics, plotting notebooks, and code that supplies inputs to a panel.

### Main figures

| Figure | Panel | Plot or analysis | Notebook or source |
| --- | --- | --- | --- |
| 1 | A | Generative state, firing rates, and observed spike waveforms | Schematic |
| 1 | B | C3PO architecture and context inference | Schematic. Implementation in [model.py](https://github.com/samuelbray32/c3po/blob/main/src/c3po/model/model.py) |
| 1 | C | Contrastive scoring and training gradients | Schematic. Related score visualization in [contrastive_geometric_interpretation.ipynb](Fig_1_2/contrastive_geometric_interpretation.ipynb) |
| 1 | D–E | Tuning reconstruction and assembly weights | Schematics |
| 2 | A | Synthetic latent phase, tuning curves, spike raster, and waveform features | [periodic_toy_model_non_coding_units_ANALYZE.ipynb](Fig_1_2/periodic_toy_model_non_coding_units_ANALYZE.ipynb) |
| 2 | B–D | Context PCA, example traces, and phase-conditioned context | [periodic_toy_model_non_coding_units_ANALYZE.ipynb](Fig_1_2/periodic_toy_model_non_coding_units_ANALYZE.ipynb) |
| 2 | E | Ground-truth and inferred neuronal tuning curves | [periodic_toy_model_non_coding_units_ANALYZE.ipynb](Fig_1_2/periodic_toy_model_non_coding_units_ANALYZE.ipynb) |
| 2 | F–G | Mark embeddings colored by peak waveform channel or preferred phase | [periodic_toy_model_non_coding_units_ANALYZE.ipynb](Fig_1_2/periodic_toy_model_non_coding_units_ANALYZE.ipynb) |
| 2 | H–J | Example neurons' tuning curves, waveforms, and mark embeddings | [periodic_toy_model_non_coding_units_ANALYZE.ipynb](Fig_1_2/periodic_toy_model_non_coding_units_ANALYZE.ipynb) |
| 3 | A–B | MEC recording, open-field task, and candidate generative variables | Schematics |
| 3 | C–E | Example context traces, theta and head-direction tuning, and spatial maps | [mec_analyze_embedding.ipynb](Fig_3/mec_analyze_embedding.ipynb) |
| 3 | F–H | Head-direction context plane, neuronal embeddings, and ordered tuning curves | [mec_analyze_embedding.ipynb](Fig_3/mec_analyze_embedding.ipynb) |
| 3 | I–K | Example head-direction decoding, error and gain by theta phase, and power spectra | [mec_analyze_headDecoding.ipynb](Fig_3/mec_analyze_headDecoding.ipynb) |
| 4 | A | CA1 recording, linear track, and candidate generative variables | Schematic |
| 4 | B | Direction-specific spatial context maps and squared correlation ratios | [rat_hpc_pc_analysis.ipynb](Fig_4/rat_hpc_pc_analysis.ipynb) |
| 4 | C | Position dependence of PC6 context and clusterless event rates | [rat_hpc_decoding.ipynb](Fig_4/rat_hpc_decoding.ipynb) |
| 4 | D | Theta-phase dependence of PC3 context and clusterless event rates | [rat_hpc_pc_analysis.ipynb](Fig_4/rat_hpc_pc_analysis.ipynb) |
| 4 | E–H | Example traversal with clusterless events, context, theta LFP, position posteriors, and decoding error | [rat_hpc_decoding.ipynb](Fig_4/rat_hpc_decoding.ipynb) |
| 4 | I | Ripple events and matched running trials | [rat_hpc_decoding.ipynb](Fig_4/rat_hpc_decoding.ipynb) |
| 5 | A–B | mPFC recording, foraging task, and trial events | Schematics |
| 5 | C | Example head speed and context traces | [rat_mpfc_sample_trace.ipynb](Fig_5_6/rat_mpfc_sample_trace.ipynb) |
| 5 | D–E | Event-aligned context, embedding loadings, and ordered firing rates | [rat_mpfc_alligned_response.ipynb](Fig_5_6/rat_mpfc_alligned_response.ipynb) |
| 5 | F–H | Slow context dynamics, motivation angle, and behavioral measures across epochs | [rat_mpfc_motivation.ipynb](Fig_5_6/rat_mpfc_motivation.ipynb), using the hybrid model |
| 5 | I–J | Example cross-correlograms and embeddings colored by putative cell type | [rat_mpfc_cellType.ipynb](Fig_5_6/rat_mpfc_cellType.ipynb) |
| 5 | K–L | Firing-rate correlations with motivation and epoch heatmaps by putative cell type | [rat_mpfc_motivation.ipynb](Fig_5_6/rat_mpfc_motivation.ipynb), using the hybrid model |
| 6 | A–B | Candidate mPFC generative variables and motivation-by-outcome interaction | Schematics |
| 6 | C | PC7 and PC8 heatmaps by motivation and preceding reward outcome | [rat_mpfc_motivation_x_reward.ipynb](Fig_5_6/rat_mpfc_motivation_x_reward.ipynb) |
| 6 | D | Neuron 451 rasters ordered by time or motivation and split by preceding outcome | [rat_mpfc_motivation_x_progression.ipynb](Fig_5_6/rat_mpfc_motivation_x_progression.ipynb) |
| 6 | E | Example neuronal firing rates by motivation and preceding outcome | [rat_mpfc_motivation_x_reward.ipynb](Fig_5_6/rat_mpfc_motivation_x_reward.ipynb) |
| 7 | A | DANDI retrieval and training workflow | Schematic. Workflow implemented in [dandiset_train_loop.py](Fig_7/dandiset_train_loop.py) and [train_from_dandi.py](https://github.com/samuelbray32/c3po/blob/main/tutorial/train_from_dandi.py) |
| 7 | B | Training time versus spike event count | [inspect_model__METRICS.ipynb](Fig_7/inspect_model__METRICS.ipynb) |
| 7 | C | Trial-aligned context during attempted finger movements | [inspect_model_FINGER_BCI.ipynb](Fig_7/inspect_model_FINGER_BCI.ipynb) |
| 7 | D | Context aligned to Ready, Set, and Go | [inspect_model_READYSETGO.ipynb](Fig_7/inspect_model_READYSETGO.ipynb) |
| 7 | E | Stimulation-aligned 3D context trajectories by current amplitude | [inspect_model_STIMULATION.ipynb](Fig_7/inspect_model_STIMULATION.ipynb) |

### Supplementary figures

| Figure | Panel | Plot or analysis | Notebook or source |
| --- | --- | --- | --- |
| S1 | A–C | Sorted, clusterless, and hybrid mark encoders | Schematics. Implementation in [encoder.py](https://github.com/samuelbray32/c3po/blob/main/src/c3po/model/encoder.py) |
| S2 | A | Phase-conditioned context PCs across model dimensionalities | [analyze_dimensionality.ipynb](Fig_1_2/analyze_dimensionality.ipynb) |
| S2 | B–C | Context power spectra and peak frequencies | [periodic_toy_model_non_coding_units_ANALYZE.ipynb](Fig_1_2/periodic_toy_model_non_coding_units_ANALYZE.ipynb) |
| S3 | A–B | Macaque S1 recording and active/passive movement conditions | Schematics |
| S3 | C–D | Active/passive context trajectories and decoded hand paths | [monkey_s1_reach_ANALYZE.ipynb](Fig_S3/monkey_s1_reach_ANALYZE.ipynb) |
| S3 | E | Reach-direction decoding confusion matrix | [monkey_s1_decoding.ipynb](Fig_S3/monkey_s1_decoding.ipynb) |
| S3 | F | Kinematic features projected onto context (top) and feature decoding performance (bottom) | [monkey_s1_decoding.ipynb](Fig_S3/monkey_s1_decoding.ipynb) (top), [monkey_s1_reach_ANALYZE.ipynb](Fig_S3/monkey_s1_reach_ANALYZE.ipynb) (bottom) |
| S4 | A–B | Head-direction tuning strength versus embedding magnitude, and theta-colored context | [mec_analyze_embedding.ipynb](Fig_3/mec_analyze_embedding.ipynb) |
| S4 | C–D | Theta-phase MUA by shank depth, and neuronal embeddings by shank and depth | [mec_analyze_embedding_thetaphase.ipynb](Fig_3/mec_analyze_embedding_thetaphase.ipynb) |
| S5 | A–D | CEBRA latent traces, feature-conditioned PCs, spatial maps, and head-direction plane | CEBRA generator unverified. Corresponding C3PO analysis in [mec_analyze_embedding.ipynb](Fig_3/mec_analyze_embedding.ipynb) |
| S5 | E–F | CEBRA head-direction decoding traces and power spectra | CEBRA generator unverified. Corresponding C3PO analysis in [mec_analyze_headDecoding.ipynb](Fig_3/mec_analyze_headDecoding.ipynb) |
| S6 | A–D | LFADS factor traces, feature-conditioned PCs, spatial maps, and head-direction plane | LFADS generator unverified. Corresponding C3PO analysis in [mec_analyze_embedding.ipynb](Fig_3/mec_analyze_embedding.ipynb) |
| S6 | E–F | LFADS head-direction decoding traces and power spectra | LFADS generator unverified. Corresponding C3PO analysis in [mec_analyze_headDecoding.ipynb](Fig_3/mec_analyze_headDecoding.ipynb) |
| S7 | All | Head-direction error boxplots and blocked permutation tests | [mec_compare_head_decode.ipynb](Fig_3/mec_compare_head_decode.ipynb), using precomputed decoder-error files |
| S8 | A | Context and theta LFP power spectra | [rat_hpc_pc_analysis.ipynb](Fig_4/rat_hpc_pc_analysis.ipynb) |
| S8 | B–D | Zoomed traversal, decoding error spectrum, and error by theta phase | [rat_hpc_decoding.ipynb](Fig_4/rat_hpc_decoding.ipynb) |
| S8 | E–F | Context and clusterless event phase precession | [rat_hpc_decoding.ipynb](Fig_4/rat_hpc_decoding.ipynb) |
| S9 | A–C | Slow context trajectories for hybrid, clusterless, and sorted-spike models | [rat_mpfc_motivation.ipynb](Fig_5_6/rat_mpfc_motivation.ipynb), run separately for each model |
| S9 | D | Within-window variance of the motivation angle across models | [rat_mpfc_motivation.ipynb](Fig_5_6/rat_mpfc_motivation.ipynb) exports the per-model variance samples. Final boxplot and permutation-test code unverified |

For Fig. 5F–H and K–L, select `mpfc_HYBRID` in the motivation notebook. For Fig. S9A–D, select `mpfc_HYBRID`, `mpfc`, or `mpfc sorted spikes` for the hybrid, clusterless, or sorted-spike model, respectively.
