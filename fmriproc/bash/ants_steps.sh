#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

# ── t1_s04: affine T1 → MNI ─────────────────────────────────────────────

ants_t1_to_mni_affine() {
  local moving="$1"     # t1_s03_unifize.nii.gz
  local fixed="$2"      # MNI template
  local output="$3"     # anat/t1_s04_space_mni152_affine.nii.gz
  local matrix="$4"     # reg/ants_t1_to_mni_0GenericAffine.mat
  local prefix="$5"     # reg/ants_t1_to_mni_  (ANTs output prefix)
  local mode="$6"

  require_file "${moving}"; require_file "${fixed}"; require_mode "${mode}"
  make_parent_dir "${output}"; mkdir -p "$(dirname "${prefix}")"

  antsRegistrationSyN.sh -d 3 -t 'a' -f "${fixed}" -m "${moving}" -o "${prefix}"

  # Copy ANTs outputs to pipeline-expected locations.
  cp "${prefix}Warped.nii.gz"        "${output}"
  cp "${prefix}0GenericAffine.mat"   "${matrix}"
}

# ── t1_s05: full (rigid+affine+nonlinear) T1 → MNI ──────────────────────

ants_t1_to_mni_nonlinear() {
  local moving="$1"     # t1_s03_unifize.nii.gz (original T1, not s04 output)
  local fixed="$2"      # MNI template
  local output="$3"     # anat/t1_s05_space_mni152_nonlinear.nii.gz
  local matrix="$4"     # reg/ants_t1_to_mni_0GenericAffine.mat
  local warp="$5"       # reg/ants_t1_to_mni_1Warp.nii.gz
  local prefix="$6"     # reg/ants_t1_to_mni_
  local mode="$7"

  require_file "${moving}"; require_file "${fixed}"; require_mode "${mode}"
  make_parent_dir "${output}"; mkdir -p "$(dirname "${prefix}")"

  antsRegistrationSyN.sh -d 3 -t 's' -f "${fixed}" -m "${moving}" -o "${prefix}"

  cp "${prefix}Warped.nii.gz"        "${output}"
  cp "${prefix}0GenericAffine.mat"   "${matrix}"
  cp "${prefix}1Warp.nii.gz"         "${warp}"
}

# ── bold_s06: affine EPI → T1 ────────────────────────────────────────────

ants_epi_to_t1_affine() {
  local epi="$1"        # bold_s05_epi_ref_unifize.nii.gz (EPI ref volume)
  local t1="$2"         # t1_s03_unifize.nii.gz
  local output="$3"     # bold_s06_epi_to_anat.nii.gz
  local matrix="$4"     # reg/ants_epi_to_t1_0GenericAffine.mat
  local prefix="$5"     # reg/ants_epi_to_t1_
  local mode="$6"
  local marker="$7"     # bold_s06_epi_to_anat.done

  require_file "${epi}"; require_file "${t1}"; require_mode "${mode}"
  make_parent_dir "${output}"; make_parent_dir "${marker}"; mkdir -p "$(dirname "${prefix}")"

  antsRegistrationSyN.sh -d 3 -t 'a' -f "${t1}" -m "${epi}" -o "${prefix}"

  cp "${prefix}Warped.nii.gz"        "${output}"
  cp "${prefix}0GenericAffine.mat"   "${matrix}"
  touch "${marker}"
}

# ── bold_s07: bring T1 masks to EPI space (ANTs inverse transform) ────────

ants_masks_to_epi() {
  local mask_dir="$1"       # masks/  (source T1 masks)
  local epi_to_t1_mat="$2"  # reg/ants_epi_to_t1_0GenericAffine.mat
  local master_epi="$3"     # bold_s05_epi_ref_unifize.nii.gz (3D EPI reference)
  local output_dir="$4"     # masks/  (destination for _to_epi masks)
  local mode="$5"
  local marker="$6"

  require_dir "${mask_dir}"; require_file "${epi_to_t1_mat}"; require_file "${master_epi}"
  require_mode "${mode}"; mkdir -p "${output_dir}"

  for mask_name in t1_brain_mask t1_wm_mask t1_csf_mask; do
    local mask_file="${mask_dir}/${mask_name}.nii.gz"
    if [[ -f "${mask_file}" ]]; then
      antsApplyTransforms -d 3 -i "${mask_file}" -o "${output_dir}/${mask_name}_to_epi.nii.gz" \
        -r "${master_epi}" -n NearestNeighbor \
        -t "[${epi_to_t1_mat},1]"
    fi
  done
  touch "${marker}"
}

# ── bold_s10: apply all transforms (EPI→T1→MNI) to 4D BOLD ──────────────

ants_apply_epi_to_mni() {
  local input="$1"          # bold_s09_grand_mean_scaling_10000.nii.gz (4D)
  local template="$2"       # MNI152_T1_3mm_brain.nii.gz
  local output="$3"         # bold_s10_space_mni152.nii.gz
  local epi_to_t1_mat="$4"  # reg/ants_epi_to_t1_0GenericAffine.mat
  local t1_to_mni_warp="$5" # reg/ants_t1_to_mni_1Warp.nii.gz
  local t1_to_mni_mat="$6"  # reg/ants_t1_to_mni_0GenericAffine.mat
  local epi_mask="$7"       # masks/t1_brain_mask_to_epi.nii.gz
  local mode="$8"

  require_file "${input}"; require_file "${template}"
  require_file "${epi_to_t1_mat}"; require_file "${t1_to_mni_warp}"; require_file "${t1_to_mni_mat}"
  require_file "${epi_mask}"
  require_mode "${mode}"
  make_parent_dir "${output}"

  local masked_input
  masked_input="$(dirname "${output}")/tmp_epi_masked_before_mni.nii.gz"
  3dcalc -a "${input}" -b "${epi_mask}" -expr 'a*b' -prefix "${masked_input}" -overwrite

  # Build a composite displacement field from the 3 ANTs transforms,
  # then apply with AFNI's 3dNwarpApply (memory-efficient for 4D).
  local warp_field
  warp_field="$(dirname "${output}")/tmp_epi_to_mni_composite.nii.gz"

  antsApplyTransforms -d 3 -o ["${warp_field}",1] \
    -r "${template}" \
    -t "${t1_to_mni_warp}" -t "${t1_to_mni_mat}" -t "${epi_to_t1_mat}"

  3dNwarpApply -source "${masked_input}" -nwarp "${warp_field}" \
    -master "${template}" -prefix "${output}"

  rm -f "${masked_input}" "${warp_field}"
}

# When called directly, dispatch first argument as function name.
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  dispatch "$@"
fi
