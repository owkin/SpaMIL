"""Transform config to transform visium counts data."""

from __future__ import annotations

from ml_collections import ConfigDict

import spamil.data.transforms.visium_deconv_transform


def set_transform_config(config):
    """Set up the config to transform RNAseq data."""
    config.transform = ConfigDict()
    config.transform.train_transform_fn = {
        "_target_": spamil.data.transforms.visium_deconv_transform.train_visium_deconv_transforms,
        "_partial_": True,
        "pca_dim": None,
        "exclude_categories": [],  # default is exclude none
        "use_fractions": False,
    }
