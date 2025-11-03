"""Base class for data transforms."""

import pathlib
from typing import Any, Iterable, Mapping, Optional, Union

import dill
import numpy as np
from ml_collections import config_dict


#
# Types (base)
#
Array = np.ndarray
ArrayTree = Union[Array, Iterable["ArrayTree"], Mapping[Any, "ArrayTree"]]


#
# Base Transform
#
class BaseTransform:
    """Base transform class.

    Defines batch transformations.

    The :code:`transform` method may include feature names into the transformed batch
    by setting :code:`batch["feature_names"]`. This can be useful in certain situations:
    for instance, when we perform feature selection and would like to inspect the names of
    the selected features (e.g., names of genes that were kept).
    """

    def transform(self, batch: ArrayTree):
        """Transform the data."""
        return batch

    def save(self, filename: Union[pathlib.Path, str]) -> pathlib.Path:
        """Save transform."""
        filename = pathlib.Path(filename).with_suffix(".pickle")
        filename.parent.mkdir(parents=True, exist_ok=True)
        with filename.open("wb") as ff:
            dill.dump(self, ff)
        return filename

    @classmethod
    def load(
        cls,
        filename: Union[pathlib.Path, str],
        config: Optional[config_dict.ConfigDict] = None,
    ):
        """Load transform."""
        del cls, config
        filename = pathlib.Path(filename).with_suffix(".pickle")
        with filename.open("rb") as ff:
            obj = dill.load(ff)
        return obj
