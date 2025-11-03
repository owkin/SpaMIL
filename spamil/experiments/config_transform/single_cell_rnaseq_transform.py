"""Transform config to transform single cell RNAseq data."""

from __future__ import annotations

from ml_collections import config_dict

import spamil.data.transforms.single_cell_rnaseq_transform


def set_transform_config(config: config_dict.ConfigDict):
    """Set up the config to transform RNAseq data."""
    config.transform = config_dict.ConfigDict()

    config.transform.train_transform_fn = config_dict.ConfigDict(
        {
            "_target_": spamil.data.transforms.single_cell_rnaseq_transform.train_single_cell_rnaseq_transforms,
            "_partial_": True,
            "rpz_dim": 100,
            "rpz_method": "scvi",
            "n_genes": 150,
            "select_based_on": "pseudobulk_variance",
            "gene_list_to_keep": "",
            "gene_list_to_exclude": "",
            "scvi_model_kwargs": {
                "n_layers": 1,  # default: 1
                "dropout_rate": 0.1,  # default: 0.1
            },
            "scvi_train_kwargs": {
                "max_epochs": 10,
                "early_stopping": False,  # default: False
                "batch_size": 4096,  # default: 128
            },
        }
    )
