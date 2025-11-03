"""File for the trainer config of the MabMIL distillation class in the OS/PFS survival task."""

import torch.nn
from ml_collections import config_dict

import spamil.utils.cindex
import spamil.utils.collate
from spamil.models.mabmil_distillation import MabMILDistillation, MabMILDistillationLoss
from spamil.utils.constants import HISTO, VISIUM_COUNTS


def set_trainer_config(
    config: config_dict.ConfigDict,
):
    """Config for the MabMIL distillation model."""
    config.trainer = config_dict.ConfigDict(
        {
            "_target_": "", # Add your torch trainer module
            "optimizer_constructor": {
                "_target_": torch.optim.Adam,
                "_partial_": True,
                "lr": 0.001,
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
                    "_target_": MabMILDistillation,
                    "_partial_": True,
                    "out_features": 1,
                    "infer_with_histo": True,
                    "histo_feature_name": HISTO,
                    "visium_feature_name": VISIUM_COUNTS,
                    "task": "OS",
                    "d_model_attention": 128,
                    "temperature": 1.0,
                    "tiles_mlp_hidden_visium": None,
                    "tiles_mlp_hidden_histo": [256],
                    "mlp_hidden": [128, 64],
                },
            },
            "per_modality_collate_fns": {
                VISIUM_COUNTS: {
                    "_target_": spamil.utils.collate.pad_collate_fn,
                    "_partial_": True,
                },
                HISTO: {
                    "_target_": spamil.utils.collate.pad_collate_fn,
                    "_partial_": True,
                },
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
            "_target_": MabMILDistillationLoss,
            "_partial_": True,
            "task": "OS",
            "prediction_loss": "cox_loss",
            "train_with_histo": False,
            "detach_prediction_modality_branch": True,
        }
    )
