gpu_to_use = 9
dims_to_run = [
    # 2,4,
    8,
    # 8,12,
    # 16,20,
    # 24,32
]

import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
import pandas as pd

os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_to_use)
import jax

from c3po.model.model import C3PO
from c3po.toy_model_generators.hmm import generate_hmm_spike_train
from c3po.toy_model_generators.phase import (
    generate_periodic_spike_train,
    generate_waveform_features,
)
from c3po.analysis.analysis import figure_directory

plt.rcParams["svg.fonttype"] = "none"  # to keep fonts editable in Illustrator

def train_model(context_dim):
    # load data
    data_name = "periodic_toy_model_non_coding_units_v2"
    data_dir = Path("/stelmo/sam/c3po_datasets") / data_name

    spiking_df = pd.read_pickle(data_dir / "spiking_data.pkl")
    neuron_df = pd.read_pickle(data_dir / "neuron_data.pkl")
    metadata_df = pd.read_pickle(data_dir / "metadata.pkl")

    mark_times = spiking_df["time"].values
    mark_ids = spiking_df["unit"].values
    marks = np.array(spiking_df["mark"].to_list())

    t_max = metadata_df["t_max"].values[0]
    latent_period = metadata_df["latent_period"].values[0]
    noise_scale = metadata_df["noise_scale"].values[0]
    n_units = metadata_df["n_units"].values[0]
    n_channels = metadata_df["n_channels"].values[0]
    max_wait_update = metadata_df["max_wait_update"].values[0]

    template_waveforms = np.stack(neuron_df["template_waveform"].values)
    tuning_curves = {
        "theta_max": neuron_df["theta_max"].values,
        "sigma": neuron_df["sigma"].values,
        "amplitude": neuron_df["amplitude"].values,
    }

    t = np.arange(0, t_max, 0.001)
    theta = np.mod(t, latent_period) / latent_period * 2 * np.pi

    ############################
    ############################
    ############################
    # prep data for model
    from c3po.model.util import prep_training_data

    delta_t = np.diff(mark_times)[None, ...]
    x = marks[1:][None, ...]

    sample_length = 2000
    x_train, delta_t_train = prep_training_data(x, delta_t, sample_length)
    i = sample_length

    # Dimensionality
    latent_dim = context_dim

    # Encoder model (waveform to Z)
    # encoder_widths = [64, 32, 16, 8]
    encoder_widths = [128, 64, 32, 32]

    encoder_args = dict(encoder_model="simple", widths=encoder_widths)

    # Context model (Z to C)
    dilations = [
        1,
        2,
        4,
        8,
    ]
    kernels = [8, 8, 16, 16]
    dilations = dilations * 2
    kernels = kernels * 2
    context_args = dict(
        context_model="wavenet",
        layer_dilations=dilations,
        layer_kernel_size=kernels,
        expanded_dim=64,
        smoothing=1,
        smoothing_decay=1,
        categorical=False,
    )

    # Rate model (C to hazard function parameters)
    rate_args = dict(
        rate_model="sharedSpace",
    )

    model = C3PO(
        encoder_args,
        context_args,
        rate_args,
        distribution="poisson",
        latent_dim=latent_dim,
        context_dim=context_dim,
        n_neg_samples=8,
        predicted_sequence_length=1,
        return_embeddings_in_call=True,
    )

    rand_key = jax.random.PRNGKey(0)
    params = model.init(jax.random.PRNGKey(42), x, delta_t, rand_key)
    run_model = jax.jit(model.apply)
    x_train.shape


    ########## TRAIN
    from c3po.model.model import train_model, update_n_neg

    params, tracked_loss = train_model(
        model,
        params,
        x_train,
        delta_t_train,
        learning_rate=1e-3,  # 3e-4,
        n_epochs=1000,
        initial_batch_size=128,
        buffer_size=8,
        min_batch_size=16,  # 32,  #   16,
        max_n_neg=256,  # 512,  # 128,
        initial_n_neg=8,
        # rand_key=jax.random.PRNGKey(43),
    )

    ############### APPLY AND STORE
    from c3po.analysis.analysis import C3poAnalysis
    model_args = dict(
        encoder_args=encoder_args,
        context_args=context_args,
        rate_args=rate_args,
        distribution="poisson",
        latent_dim=latent_dim,
        context_dim=context_dim,
        n_neg_samples=4,
        predicted_sequence_length=1,
        sample_params=None,
    )
    analysis = C3poAnalysis(
        model=model,
        model_args=model_args,
        params=params,
    )

    analysis.embed_data(x, delta_t, first_mark_time=0, chunk_size=5000, delta_t_units="s")
    analysis.fit_context_pca()
    t_interp = np.arange(analysis.t[0], analysis.t[-1], 0.001)
    analysis.interpolate_context(t_interp)
    analysis.embed_context_pca()

    model_tag = f"dim_sweep_{context_dim}d"
    model_name = f"{data_name}_{model_tag}"

    from flax import serialization
    from c3po.tables.dev_tables import C3POStorage

    C3POStorage()
    insert_key = {
        "model_name": model_name,
        "encoder_args": encoder_args,
        "context_args": context_args,
        "rate_args": rate_args,
        "latent_dim": latent_dim,
        "context_dim": context_dim,
        "learned_params": serialization.to_bytes(params),
        "input_shape": np.array(x.shape[-1]),
    }

    C3POStorage().insert1(insert_key)
    analysis.save_embedding(f"/stelmo/sam/c3po_results/{model_name}_embedding.npz")

    return


if __name__ == "__main__":
    for context_dim in dims_to_run:
        train_model(context_dim)

