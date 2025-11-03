"""Transform config to transform bulk RNAseq data."""

from __future__ import annotations

from ml_collections import config_dict

import spamil.data.transforms.bulk_rnaseq_transform


def set_transform_config(
    config: config_dict.ConfigDict,
):
    """Set up the config to transform RNAseq data."""
    config.transform = config_dict.ConfigDict()

    config.transform.train_transform_fn = config_dict.ConfigDict(
        {
            "_target_": spamil.data.transforms.bulk_rnaseq_transform.train_rnaseq_transforms,
            "_partial_": True,
            "max_genes": 3000,
            "filtering_method": "variance",
            "scaling_method": "log-min_max",
            "gene_list": "",  # path to gene list
        }
    )
