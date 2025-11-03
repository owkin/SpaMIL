"""Constant file."""

VISIUM_COUNTS = "visium_counts"
VISIUM_DECONV = "visium_deconv"
BULK_RNASEQ = "rnaseq"
SINGLE_CELL_RNASEQ = "single_cell_rnaseq"
HISTO = "histo"
HISTO_ALIGNED = "histo_aligned"  # histo feature tiles on visium spots

NORMALIZATION = {
    "GBM_UKER": {
        BULK_RNASEQ: {"raw", "tpm"},
        VISIUM_COUNTS: {"raw_count", "norm_count", "log_norm_count", "SCTransform"},
        SINGLE_CELL_RNASEQ: {"ambient_rna_removed", "LogNormalize"},
        VISIUM_DECONV: {"raw_count", "norm_count", "log_norm_count", "SCTransform"},
    },
    "GBM_CHUV": {
        BULK_RNASEQ: {"raw", "tpm"},
        VISIUM_COUNTS: {"raw_count", "norm_count", "log_norm_count", "SCTransform"},
        SINGLE_CELL_RNASEQ: {"ambient_rna_removed", "LogNormalize"},
        VISIUM_DECONV: {"raw_count", "norm_count", "log_norm_count", "SCTransform"},
    },
}

ENDPOINTS = {
    "GBM_UKER": {"OS"},
    "GBM_CHUV": {"OS"},
}
PATIENT_ID_CLINICAL = "Tumour block ID for MOSAIC"
PATIENT_ID = "sample_id"  # index column to align dataframes
