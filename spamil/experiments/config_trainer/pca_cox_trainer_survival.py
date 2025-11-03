"""File for the trainer config of the PCACox class for survival tasks."""

from __future__ import annotations

from ml_collections import config_dict

import spamil.utils.cindex


def set_trainer_config(
    config: config_dict.ConfigDict,
):
    """Config for the pca cox model."""
    config.trainer = config_dict.ConfigDict(
        {
            # Add your module that computes both `sklearn.decomposition.PCA`,
            # and `lifelines.fitters.coxph_fitter.CoxPHFitter`
            "_target_": "",
            "task": "OS",
            "pca_dims": 5,
            "penalizer": 0.001,
            "l1_ratio": 0.5,
            "aggregation_method": "ensembling",
        }
    )

    config.trainer.metrics = config_dict.ConfigDict(
        {
            "c_index": {
                "_target_": spamil.utils.cindex.compute_cindex,
                "_partial_": True,
            },
        }
    )
