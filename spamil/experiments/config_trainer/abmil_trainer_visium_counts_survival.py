"""File for the trainer config of the abMIL class in the OS/PFS survival task."""

import torch.nn
from ml_collections import config_dict

import spamil.models.utils
import spamil.utils.cindex
import spamil.utils.collate
from spamil.models.abmil import AbMIL
from spamil.utils.constants import VISIUM_COUNTS


def set_trainer_config(
    config: config_dict.ConfigDict,
):
    """Config for the abMIL model."""
    config.trainer = config_dict.ConfigDict(
        {
            "_target_": "", # Add your torch trainer module
            "optimizer_constructor": {
                "_target_": torch.optim.Adam,
                "_partial_": True,
                "lr": 0.0003,
                "weight_decay": 0,
            },
            "batch_size": 64,
            "n_epochs": 30,
            "eval_every": 2,
            "device": "cuda:0",
            "build_model_fn": {
                "_target_": "", # Add the module that instantiates your loss and prediction modules
                "_partial_": True,
                "module_constructor": {
                    "_target_": AbMIL,
                    "_partial_": True,
                    "out_features": 1,
                    "modality": VISIUM_COUNTS,
                    "task": "OS",
                    "d_model_attention": 256,
                    "temperature": 1.0,
                    "tiles_mlp_hidden": None,
                    "mlp_hidden": [128, 64],
                },
            },
            "per_modality_collate_fns": {
                VISIUM_COUNTS: {
                    "_target_": spamil.utils.collate.pad_collate_fn,
                    "_partial_": True,
                }
            },
        }
    )
    config.trainer.metrics_per_task = config_dict.ConfigDict(
        {
            "OS": {
                "c_index": {
                    "_target_": spamil.utils.cindex.compute_cindex,
                    "_partial_": True,
                },
            }
        }
    )
    config.trainer.build_model_fn.module_constructor.mlp_activation = torch.nn.Sigmoid()
    config.trainer.build_model_fn.loss_constructor = config_dict.ConfigDict(
        {
            "_target_": spamil.models.utils.CoxLoss,
            "_partial_": True,
            "task": "OS",
        }
    )
