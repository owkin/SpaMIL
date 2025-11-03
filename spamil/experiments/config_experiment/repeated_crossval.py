"""Experiment config to set up the repeated cross validation."""

from ml_collections import config_dict

from spamil.utils.default_split import default_split


def set_experiment_config(
    config: config_dict.ConfigDict,
):
    """Set up the config to enable repeated cross validation."""
    config.experiment = config_dict.ConfigDict()
    config.experiment.seed = 42
    config.experiment.save_trainer = True
    config.experiment.save_transform = True

    config.experiment.data_split_method = {
        "_target_": default_split,
        "_partial_": True,
        "n_splits": 5,
        "n_repeats": 10,
        "stratify_for_survival_or_binary_classification": True,
    }
