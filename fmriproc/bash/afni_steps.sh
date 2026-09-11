#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

afni_t1_deoblique() {
  local input="$1"; local output="$2"; local mode="$3"
  require_file "${input}"; require_mode "${mode}"; make_parent_dir "${output}"
  3dWarp -oblique2card -prefix "${output}" -newgrid 1.0 "${input}"
}

afni_t1_skullstrip() {
  local input="$1"; local output="$2"; local mode="$3"
  require_file "${input}"; require_mode "${mode}"; make_parent_dir "${output}"
  if [[ "${mode}" == "tested" ]]; then
    3dSkullStrip -input "${input}" -prefix "${output}" -push_to_edge
  else
    3dSkullStrip -input "${input}" -prefix "${output}"
  fi
}

afni_t1_unifize() {
  local input="$1"; local output="$2"; local mode="$3"
  require_file "${input}"; require_mode "${mode}"; make_parent_dir "${output}"
  3dUnifize -input "${input}" -prefix "${output}"
}

afni_t1_to_mni_affine() {
  local input="$1"; local template="$2"; local output="$3"; local matrix="$4"; local mode="$5"
  require_file "${input}"; require_file "${template}"; require_mode "${mode}"; make_parent_dir "${output}"; make_parent_dir "${matrix}"
  if [[ "${mode}" == "tested" ]]; then
    3dAllineate -prefix "${output}" -base "${template}" -source "${input}" -twopass -cost lpa \
      -1Dmatrix_save "${matrix}" -autoweight -source_automask -fineblur 3 -cmass
  else
    3dAllineate -prefix "${output}" -base "${template}" -source "${input}" -1Dmatrix_save "${matrix}"
  fi
}

afni_t1_to_mni_nonlinear() {
  local input="$1"; local template="$2"; local output="$3"; local mode="$4"; local warp_output="${5:-}"
  require_file "${input}"; require_file "${template}"; require_mode "${mode}"; make_parent_dir "${output}"
  if [[ "${mode}" == "tested" ]]; then
    3dQwarp -prefix "${output}" -blur 0 3 -base "${template}" -source "${input}" -overwrite
  else
    3dQwarp -prefix "${output}" -base "${template}" -source "${input}" -overwrite
  fi
  if [[ -n "${warp_output}" ]]; then
    make_parent_dir "${warp_output}"
    # 3dQwarp creates {prefix}_WARP.nii.gz alongside the output, but the _WARP suffix
    # is appended to the full filename.  The warp lives at dirname(output)/basename(output)_WARP.nii.gz
    local warp_src
    warp_src="$(dirname "${output}")/$(basename "${output}" .nii.gz)_WARP.nii.gz"
    if [[ -f "${warp_src}" ]]; then
      cp "${warp_src}" "${warp_output}"
    fi
  fi
}

afni_t1_segmentation() {
  local input="$1"; local output_dir="$2"; local mode="$3"; local marker="$4"
  require_file "${input}"; require_mode "${mode}"; mkdir -p "${output_dir}"
  # 3dSeg creates a Segsy/ subdirectory in the CWD, so cd to the parent of output_dir.
  local abs_input; abs_input="$(cd "$(dirname "${input}")" && pwd)/$(basename "${input}")"
  local abs_out_parent; abs_out_parent="$(mkdir -p "${output_dir}" && cd "$(dirname "${output_dir}")" && pwd)"
  (
    cd "${abs_out_parent}"
    3dSeg -anat "${abs_input}" -mask AUTO -classes 'CSF ; GM ; WM' \
      -bias_classes 'GM ; WM' -bias_fwhm 25 -mixfrac UNI -main_N 5 -blur_meth BFT
  )
  touch "${marker}"
}

afni_t1_masks() {
  local input="$1"; local classes="$2"; local mask_dir="$3"; local mode="$4"; local marker="$5"
  require_file "${input}"; require_mode "${mode}"; mkdir -p "${mask_dir}"
  3dAutomask -prefix "${mask_dir}/t1_brain_mask.nii.gz" -dilate 1 -overwrite "${input}"
  # 3dSeg outputs a single sub-brick with integer labels:
  #   0=background, 1=CSF, 2=GM, 3=WM (matching -classes 'CSF ; GM ; WM')
  3dcalc -a "${classes}" -expr 'equals(a,3)' -prefix "${mask_dir}/t1_wm_mask.nii.gz" -overwrite
  3dcalc -a "${classes}" -expr 'equals(a,1)' -prefix "${mask_dir}/t1_csf_mask.nii.gz" -overwrite
  touch "${marker}"
}

afni_bold_drop_dummy() {
  local input="$1"; local output="$2"; local dummy="$3"; local mode="$4"
  require_file "${input}"; require_mode "${mode}"; make_parent_dir "${output}"
  3dcalc -a "${input}[${dummy}..$]" -expr 'a' -prefix "${output}"
}

afni_bold_deoblique() {
  local input="$1"; local output="$2"; local mode="$3"
  require_file "${input}"; require_mode "${mode}"; make_parent_dir "${output}"
  local grid
  grid="$(3dinfo -ad3 "${input}" | awk '{print $1}')"
  3dWarp -oblique2card -prefix "${output}" -newgrid "${grid}" "${input}"
}

afni_bold_motion_correction() {
  local input="$1"; local output="$2"; local motion_1d="$3"; local motion_mat="$4"; local mode="$5"
  require_file "${input}"; require_mode "${mode}"; make_parent_dir "${output}"; make_parent_dir "${motion_1d}"; make_parent_dir "${motion_mat}"
  if [[ "${mode}" == "tested" ]]; then
    3dvolreg -prefix "${output}" -verbose -zpad 1 -Fourier -base 0 -twopass \
      -1Dfile "${motion_1d}" -1Dmatrix_save "${motion_mat}" "${input}"
  else
    3dvolreg -prefix "${output}" -base 0 -1Dfile "${motion_1d}" "${input}"
  fi
}

afni_bold_slice_timing() {
  local input="$1"; local output="$2"; local tr="$3"; local pattern="$4"; local mode="$5"
  require_file "${input}"; require_mode "${mode}"; make_parent_dir "${output}"
  3dTshift -verbose -TR "${tr}" -tpattern "${pattern}" -prefix "${output}" "${input}"
}

afni_bold_epi_ref_unifize() {
  local input="$1"; local bucket="$2"; local output="$3"; local mode="$4"
  require_file "${input}"; require_mode "${mode}"; make_parent_dir "${bucket}"; make_parent_dir "${output}"
  3dbucket -prefix "${bucket}" "${input}[0]"
  3dUnifize -prefix "${output}" "${bucket}"
}

afni_bold_epi_to_anat() {
  local epi="$1"; local anat="$2"; local output_dir="$3"; local mode="$4"; local marker="$5"
  require_file "${epi}"; require_file "${anat}"; require_mode "${mode}"; mkdir -p "${output_dir}"
  align_epi_anat.py -epi2anat -anat "${anat}" -epi "${epi}" -epi_base 0 \
    -partial_axial -master_epi "${anat}" -epi_strip 3dAutomask -suffix _align -big_move \
    -output_dir "${output_dir}"
  touch "${marker}"
}

afni_bold_masks_to_epi() {
  local mask_dir="$1"; local matrix="$2"; local master="$3"; local output_dir="$4"; local mode="$5"; local marker="$6"
  require_dir "${mask_dir}"; require_file "${matrix}"; require_file "${master}"; require_mode "${mode}"; mkdir -p "${output_dir}"

  # Invert epi→anat matrix to get anat→epi.
  local inv_matrix
  inv_matrix="${output_dir}/anat2epi.aff12.1D"
  cat_matvec -ONELINE "${matrix}" -I > "${inv_matrix}"

  for mask_name in t1_brain_mask t1_wm_mask t1_csf_mask; do
    local mask_file="${mask_dir}/${mask_name}.nii.gz"
    if [[ -f "${mask_file}" ]]; then
      3dAllineate -input "${mask_file}" -1Dmatrix_apply "${inv_matrix}" \
        -interp linear -final linear -master "${master}" \
        -prefix "${output_dir}/${mask_name}_to_epi.nii.gz"
    fi
  done
  touch "${marker}"
}

afni_bold_nuisance_regression() {
  local input="$1"; local output="$2"; local mask="$3"; local csf_mask="$4"; local wm_mask="$5"; local nuisance_1d="$6"; local tr="$7"; local highpass="$8"; local lowpass="$9"; local mode="${10}"
  require_file "${input}"; require_file "${mask}"; require_file "${nuisance_1d}"; require_mode "${mode}"; make_parent_dir "${output}"
  local out_dir
  out_dir="$(dirname "${output}")"
  local mean_file="${out_dir}/tmp_mean_for_nuisance.nii.gz"
  local res_file="${out_dir}/tmp_nuisance_residual.nii.gz"

  3dTstat -mean -prefix "${mean_file}" -overwrite "${input}"
  3dTproject -input "${input}" -prefix "${res_file}" -mask "${mask}" \
    -bandpass "${highpass}" "${lowpass}" \
    -ort "${nuisance_1d}" -dt "${tr}"
  3dcalc -a "${res_file}" -b "${mean_file}" -c "${mask}" -expr '(a+b)*c' -prefix "${output}" -overwrite
  rm -f "${mean_file}" "${res_file}"
}

afni_bold_grand_mean_scaling() {
  local input="$1"; local mask="$2"; local output="$3"; local target="$4"; local mode="$5"
  require_file "${input}"; require_file "${mask}"; require_mode "${mode}"; make_parent_dir "${output}"
  local mean_file
  mean_file="$(dirname "${output}")/tmp_gms_mean.nii.gz"
  3dTstat -mean -mask "${mask}" -prefix "${mean_file}" "${input}"
  local grand_mean
  grand_mean="$(3dmaskave -mask "${mask}" -quiet "${mean_file}")"
  3dcalc -a "${input}" -b "${mask}" -expr "a*${target}/${grand_mean}*b" -prefix "${output}"
  rm -f "${mean_file}"
}

afni_bold_space_mni152() {
  local input="$1"; local warp="$2"; local anat_affine="$3"; local epi_affine="$4"; local epi_mask="$5"; local output="$6"; local mode="$7"
  require_file "${input}"; require_file "${warp}"; require_file "${anat_affine}"; require_file "${epi_affine}"; require_file "${epi_mask}"; require_mode "${mode}"; make_parent_dir "${output}"

  local masked_input
  masked_input="$(dirname "${output}")/tmp_epi_masked_before_mni.nii.gz"
  3dcalc -a "${input}" -b "${epi_mask}" -expr 'a*b' -prefix "${masked_input}" -overwrite

  3dNwarpApply -source "${masked_input}" -nwarp "${warp} ${anat_affine} ${epi_affine}" \
    -master WARP -prefix "${output}"

  rm -f "${masked_input}"
}

afni_bold_smooth() {
  local input="$1"; local output="$2"; local fwhm="$3"; local mode="$4"
  require_file "${input}"; require_mode "${mode}"; make_parent_dir "${output}"
  3dmerge -1blur_fwhm "${fwhm}" -doall -prefix "${output}" "${input}"
}

# When called directly, dispatch first argument as function name.
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  dispatch "$@"
fi
