"""The config from the best mabmil multimodal inference experiment in GBM."""

from pathlib import Path

from ml_collections import config_dict

from spamil.experiments.config_experiment.repeated_crossval import set_experiment_config
from spamil.experiments.config_trainer.mabmil_multimodal_inference_trainer_histo_visium_counts_survival import (
    set_trainer_config,
)
from spamil.experiments.config_transform.visium_counts_transform import (
    set_transform_config,
)


def get_config():
    """Best composed config using mabmil multimodal inference."""
    config = config_dict.ConfigDict()
    config.base_output_dir = Path("") # Add your output directory of the experiment results

    # Add your data config, setting up spatial transcriptomics and histology loading and parameters
    # set_data_config(config)

    # The data config is composed of the following parameters
    config.data = config_dict.ConfigDict()
    config.data.dataset_constructor = config_dict.ConfigDict(
        {
            "_target_": "", # Add your dataset module
            "_partial_": True,
            "visium_normalization": "raw_count",
            "histo_extractor": "h0_features",
            "aligned_tiles_spots": True,
            "n_tiles_max": 4000,
            "shuffle_tiles": True,
            "endpoint": "OS",
            "cohorts": ["GBM_CHUV", "GBM_UKER"],
            "normalization": "raw_count",
            # Primary samples from the first batch that pass QC
            "inclusion_criteria": {
                "GBM_CHUV": ["group1", "primary_only", "passes_usability"],
                "GBM_UKER": ["group1", "primary_only", "passes_usability"],
            },
        }
    )

    # Rest of the experiment configs
    set_experiment_config(config)
    config.experiment.save_trainer = True
    config.experiment.save_transform = True
    set_trainer_config(config)
    set_transform_config(config)

    config.experiment.experiment_name = "mabmil_multimodal_inference_best1"
    config.transform.train_transform_fn.n_genes = 500
    config.transform.train_transform_fn.select_based_on = "morans_i"
    config.transform.train_transform_fn.rpz_dim = 256
    config.transform.train_transform_fn.rpz_method = "pca"

    return config
