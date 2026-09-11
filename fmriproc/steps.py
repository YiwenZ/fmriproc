from __future__ import annotations

from .models import StepDefinition


ANAT_STEPS: tuple[StepDefinition, ...] = (
    StepDefinition("t1_s01_deoblique", "anat", "afni_t1_deoblique", "t1_s01_deoblique.nii.gz"),
    StepDefinition("t1_s02_skullstrip", "anat", "afni_t1_skullstrip", "t1_s02_skullstrip.nii.gz", ("t1_s01_deoblique",)),
    StepDefinition("t1_s03_unifize", "anat", "afni_t1_unifize", "t1_s03_unifize.nii.gz", ("t1_s02_skullstrip",)),
    StepDefinition("t1_s04_space_mni152_affine", "anat", "afni_t1_to_mni_affine", "t1_s04_space_mni152_affine.nii.gz", ("t1_s03_unifize",)),
    StepDefinition("t1_s05_space_mni152_nonlinear", "anat", "afni_t1_to_mni_nonlinear", "t1_s05_space_mni152_nonlinear.nii.gz", ("t1_s04_space_mni152_affine",)),
    StepDefinition("t1_s06_segmentation", "anat", "afni_t1_segmentation", "t1_s06_segmentation.done", ("t1_s03_unifize",)),
    StepDefinition("t1_s07_masks", "anat", "afni_t1_masks", "t1_s07_masks.done", ("t1_s06_segmentation",)),
)

BOLD_STEPS: tuple[StepDefinition, ...] = (
    StepDefinition("bold_s01_drop_dummy", "bold", "afni_bold_drop_dummy", "bold_s01_drop_dummy.nii.gz"),
    StepDefinition("bold_s02_deoblique", "bold", "afni_bold_deoblique", "bold_s02_deoblique.nii.gz", ("bold_s01_drop_dummy",)),
    StepDefinition("bold_s03_motion_correction", "bold", "afni_bold_motion_correction", "bold_s03_motion_correction.nii.gz", ("bold_s02_deoblique",)),
    StepDefinition("bold_s04_slice_timing", "bold", "afni_bold_slice_timing", "bold_s04_slice_timing.nii.gz", ("bold_s03_motion_correction",), optional=True),
    StepDefinition("bold_s05_epi_ref_unifize", "bold", "afni_bold_epi_ref_unifize", "bold_s05_epi_ref_unifize.nii.gz", ("bold_s03_motion_correction",)),
    StepDefinition("bold_s06_epi_to_anat", "bold", "afni_bold_epi_to_anat", "bold_s06_epi_to_anat.done", ("bold_s05_epi_ref_unifize",)),
    StepDefinition("bold_s07_masks_to_epi", "bold", "afni_bold_masks_to_epi", "bold_s07_masks_to_epi.done", ("bold_s06_epi_to_anat",)),
    StepDefinition("bold_s08_nuisance_regression", "bold", "afni_bold_nuisance_regression", "bold_s08_nuisance_regression.nii.gz", ("bold_s07_masks_to_epi",)),
    StepDefinition("bold_s09_grand_mean_scaling_10000", "bold", "afni_bold_grand_mean_scaling", "bold_s09_grand_mean_scaling_10000.nii.gz", ("bold_s08_nuisance_regression",)),
    StepDefinition("bold_s10_space_mni152", "bold", "afni_bold_space_mni152", "bold_s10_space_mni152.nii.gz", ("bold_s09_grand_mean_scaling_10000",)),
    StepDefinition("bold_s11_smooth_sm6", "bold", "afni_bold_smooth", "bold_s11_smooth_sm6.nii.gz", ("bold_s10_space_mni152",)),
)


def get_step_ids(steps: tuple[StepDefinition, ...]) -> list[str]:
    return [step.step_id for step in steps]
