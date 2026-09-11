from __future__ import annotations

from .models import StepDefinition

FSL_ANAT_STEPS: tuple[StepDefinition, ...] = (
    StepDefinition("t1_s01_reorient2std", "anat", "fsl_t1_reorient2std", "t1_s01_reorient2std.nii.gz"),
    StepDefinition("t1_s02_skullstrip_bet", "anat", "fsl_t1_skullstrip_bet", "t1_s02_skullstrip_bet.nii.gz", ("t1_s01_reorient2std",)),
    StepDefinition("t1_s03_fast_segmentation", "anat", "fsl_t1_fast_segmentation", "t1_s03_fast_segmentation.done", ("t1_s02_skullstrip_bet",)),
    StepDefinition("t1_s04_masks", "anat", "fsl_t1_masks", "t1_s04_masks.done", ("t1_s03_fast_segmentation",)),
    StepDefinition("t1_s05_space_mni152_affine", "anat", "fsl_t1_to_mni_affine", "t1_s05_space_mni152_affine.nii.gz", ("t1_s02_skullstrip_bet",)),
    StepDefinition("t1_s06_space_mni152_nonlinear", "anat", "fsl_t1_to_mni_nonlinear", "t1_s06_space_mni152_nonlinear.nii.gz", ("t1_s05_space_mni152_affine",)),
)

FSL_BOLD_STEPS: tuple[StepDefinition, ...] = (
    StepDefinition("bold_s01_reorient2std", "bold", "fsl_bold_reorient2std", "bold_s01_reorient2std.nii.gz"),
    StepDefinition("bold_s02_drop_dummy", "bold", "fsl_bold_drop_dummy", "bold_s02_drop_dummy.nii.gz", ("bold_s01_reorient2std",)),
    StepDefinition("bold_s03_ref", "bold", "fsl_bold_ref", "bold_s03_ref.nii.gz", ("bold_s02_drop_dummy",)),
    StepDefinition("bold_s04_motion_correction_mcflirt", "bold", "fsl_bold_motion_correction_mcflirt", "bold_s04_motion_correction_mcflirt.nii.gz", ("bold_s02_drop_dummy", "bold_s03_ref")),
    StepDefinition("bold_s05_slice_timing", "bold", "fsl_bold_slice_timing", "bold_s05_slice_timing.nii.gz", ("bold_s04_motion_correction_mcflirt",), optional=True),
    StepDefinition("bold_s06_epi_to_anat_bbr", "bold", "fsl_bold_epi_to_anat_bbr", "bold_s06_epi_to_anat_bbr.done", ("bold_s03_ref", "t1_s02_skullstrip_bet")),
    StepDefinition("bold_s07_masks_to_epi", "bold", "fsl_bold_masks_to_epi", "bold_s07_masks_to_epi.done", ("bold_s06_epi_to_anat_bbr", "t1_s04_masks")),
    StepDefinition("bold_s08_nuisance_regression", "bold", "fsl_bold_nuisance_regression", "bold_s08_nuisance_regression.nii.gz", ("bold_s04_motion_correction_mcflirt", "bold_s07_masks_to_epi")),
    StepDefinition("bold_s09_temporal_filtering", "bold", "fsl_bold_temporal_filtering", "bold_s09_temporal_filtering.nii.gz", ("bold_s08_nuisance_regression",)),
    StepDefinition("bold_s10_grand_mean_scaling_10000", "bold", "fsl_bold_grand_mean_scaling", "bold_s10_grand_mean_scaling_10000.nii.gz", ("bold_s09_temporal_filtering",)),
    StepDefinition("bold_s11_space_mni152", "bold", "fsl_bold_space_mni152", "bold_s11_space_mni152.nii.gz", ("bold_s10_grand_mean_scaling_10000", "bold_s06_epi_to_anat_bbr", "t1_s06_space_mni152_nonlinear")),
    StepDefinition("bold_s12_smooth_sm6", "bold", "fsl_bold_smooth", "bold_s12_smooth_sm6.nii.gz", ("bold_s11_space_mni152",)),
)
