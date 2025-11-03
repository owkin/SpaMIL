"""Transform config to transform clinical data."""

from __future__ import annotations

from ml_collections import config_dict

import spamil.data.transforms.clinical_transform


def set_transform_config(
    config: config_dict.ConfigDict,
):
    """Set up the config to transform RNAseq data."""
    config.transform = config_dict.ConfigDict()

    config.transform.train_transform_fn = config_dict.ConfigDict(
        {
            "_target_": spamil.data.transforms.clinical_transform.train_clinical_transforms,
            "_partial_": True,
        }
    )
