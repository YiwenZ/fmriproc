#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

# ── t1_s01: reorient to standard ──────────────────────────────────────────

fsl_t1_reorient2std() {
  local input="$1"; local output="$2"; local mode="$3"
  require_file "${input}"; require_mode "${mode}"; make_parent_dir "${output}"
  fslreorient2std "${input}" "${output}"
}

# ── t1_s02: skullstrip with BET ───────────────────────────────────────────

fsl_t1_skullstrip_bet() {
  local input="$1"; local output="$2"; local mode="$3"
  require_file "${input}"; require_mode "${mode}"; make_parent_dir "${output}"
  if command -v bc >/dev/null 2>&1; then
    bet "${input}" "${output}" -f 0.1 -B -R -m
  else
    echo "[WARN] bc not found; falling back to DWI-style basic BET parameters: -f 0.2 -m" >&2
    bet "${input}" "${output}" -f 0.2 -m
  fi
}

# ── t1_s03: FAST segmentation ─────────────────────────────────────────────

fsl_t1_fast_segmentation() {
  local input="$1"; local seg_dir="$2"; local mode="$3"; local marker="$4"
  require_file "${input}"; require_mode "${mode}"
  mkdir -p "${seg_dir}"
  fast -S 1 -t 1 -o "${seg_dir}/t1_seg" -n 3 -b -B -v "${input}"
  touch "${marker}"
}

# ── t1_s04: derive masks from FAST PVE outputs ────────────────────────────

fsl_t1_masks() {
  local t1_brain="$1"; local seg_dir="$2"; local mask_dir="$3"; local mode="$4"; local marker="$5"
  require_file "${t1_brain}"; require_mode "${mode}"
  mkdir -p "${mask_dir}"

  local gm="${mask_dir}/t1_gm_mask.nii.gz"
  local wm="${mask_dir}/t1_wm_mask.nii.gz"
  local csf="${mask_dir}/t1_csf_mask.nii.gz"
  local brain_mask="${mask_dir}/t1_brain_mask.nii.gz"

  # FAST pve_0=CSF, pve_1=GM, pve_2=WM (with -b flag, order may vary; use -B for no bias)
  fslmaths "${seg_dir}/t1_seg_pve_0.nii.gz" -thr 0.9 -bin "${csf}"
  fslmaths "${seg_dir}/t1_seg_pve_2.nii.gz" -thr 0.9 -bin "${wm}"
  fslmaths "${seg_dir}/t1_seg_pve_1.nii.gz" -thr 1.0 -bin "${gm}"

  # Brain mask from BET sidecar (beside t1_brain, e.g. t1_s02_skullstrip_bet_mask.nii.gz).
  local bet_mask
  bet_mask="$(dirname "${t1_brain}")/$(basename "${t1_brain}" .nii.gz)_mask.nii.gz"
  if [[ -f "${bet_mask}" ]]; then
    cp "${bet_mask}" "${brain_mask}"
  fi

  touch "${marker}"
}

# ── t1_s05: affine T1 → MNI via FLIRT ─────────────────────────────────────

fsl_t1_to_mni_affine() {
  local t1_brain="$1"; local template="$2"; local output="$3"; local matrix="$4"; local mode="$5"
  require_file "${t1_brain}"; require_file "${template}"; require_mode "${mode}"
  make_parent_dir "${output}"; mkdir -p "$(dirname "${matrix}")"
  flirt -ref "${template}" -in "${t1_brain}" -dof 12 -omat "${matrix}" -out "${output}"
}

# ── t1_s06: nonlinear T1 → MNI via FNIRT ──────────────────────────────────

fsl_t1_to_mni_nonlinear() {
  local t1_reorient="$1"; local affine_mat="$2"; local output="$3"; local warp="$4"; local coef="$5"; local mode="$6"
  require_file "${t1_reorient}"; require_file "${affine_mat}"; require_mode "${mode}"
  make_parent_dir "${output}"
  mkdir -p "$(dirname "${warp}")"; mkdir -p "$(dirname "${coef}")"

  fnirt --in="${t1_reorient}" --aff="${affine_mat}" \
    --config=T1_2_MNI152_2mm --iout="${output}" \
    --cout="${coef}" --fout="${warp}"
}

# ── bold_s01: reorient to standard ────────────────────────────────────────

fsl_bold_reorient2std() {
  local input="$1"; local output="$2"; local mode="$3"
  require_file "${input}"; require_mode "${mode}"; make_parent_dir "${output}"
  fslreorient2std "${input}" "${output}"
}

# ── bold_s02: drop dummy scans ────────────────────────────────────────────

fsl_bold_drop_dummy() {
  local input="$1"; local output="$2"; local dummy="$3"; local length_after="$4"; local mode="$5"
  require_file "${input}"; require_mode "${mode}"; make_parent_dir "${output}"
  fslroi "${input}" "${output}" "${dummy}" "${length_after}"
}

# ── bold_s03: extract reference volume ─────────────────────────────────────

fsl_bold_ref() {
  local input="$1"; local output="$2"; local mode="$3"
  require_file "${input}"; require_mode "${mode}"; make_parent_dir "${output}"
  fslroi "${input}" "${output}" 0 1
}

# ── bold_s04: motion correction with MCFLIRT ──────────────────────────────

fsl_bold_motion_correction_mcflirt() {
  local input="$1"; local output="$2"; local ref="$3"; local motion_par="$4"; local mode="$5"
  require_file "${input}"; require_file "${ref}"; require_mode "${mode}"
  make_parent_dir "${output}"; mkdir -p "$(dirname "${motion_par}")"

  mcflirt -in "${input}" -out "${output}" -reffile "${ref}" \
    -spline_final -plots -mats -rmsrel -rmsabs

  # mcflirt creates output.par; move if not at the desired path.
  if [[ -f "${output}.par" ]] && [[ "${output}.par" != "${motion_par}" ]]; then
    cp "${output}.par" "${motion_par}"
  fi
}

# ── bold_s05: slice timing correction ─────────────────────────────────────

fsl_bold_slice_timing() {
  local input="$1"; local output="$2"; local tr="$3"; local direction="$4"; local interleaved="$5"; local mode="$6"
  require_file "${input}"; require_mode "${mode}"; make_parent_dir "${output}"

  local odd_even_flag=""
  if [[ "${interleaved}" == "odd" ]]; then
    odd_even_flag="--odd"
  elif [[ "${interleaved}" == "even" ]]; then
    odd_even_flag="--down"
  fi

  slicetimer -i "${input}" -o "${output}" -r "${tr}" -d "${direction}" ${odd_even_flag}
}

# ── bold_s06: EPI → T1 with BBR ───────────────────────────────────────────

fsl_bold_epi_to_anat_bbr() {
  local epi_ref="$1"; local t1="$2"; local t1_brain="$3"; local out_prefix="$4"; local mode="$5"; local marker="$6"
  require_file "${epi_ref}"; require_file "${t1}"; require_file "${t1_brain}"; require_mode "${mode}"
  mkdir -p "$(dirname "${out_prefix}")"

  epi_reg --epi="${epi_ref}" --t1="${t1}" --t1brain="${t1_brain}" --out="${out_prefix}"
  require_file "${out_prefix}.mat"
  touch "${marker}"
}

# ── bold_s07: bring T1 masks to EPI space ──────────────────────────────────

fsl_bold_masks_to_epi() {
  local mask_dir="$1"; local epi_to_anat_mat="$2"; local bold_ref="$3"; local output_dir="$4"; local mode="$5"; local marker="$6"
  require_dir "${mask_dir}"; require_file "${epi_to_anat_mat}"; require_file "${bold_ref}"
  require_mode "${mode}"; mkdir -p "${output_dir}"

  local inv_mat="${output_dir}/fsl_epi_to_anat_bbr_inv.mat"
  convert_xfm -omat "${inv_mat}" -inverse "${epi_to_anat_mat}"

  for mask_name in t1_brain_mask t1_wm_mask t1_csf_mask; do
    local src="${mask_dir}/${mask_name}.nii.gz"
    local dest="${output_dir}/${mask_name}_to_epi.nii.gz"
    if [[ -f "${src}" ]]; then
      flirt -ref "${bold_ref}" -in "${src}" -applyxfm -init "${inv_mat}" \
        -interp nearestneighbour -out "${dest}"
    fi
  done
  touch "${marker}"
}

# ── bold_s08: nuisance regression ─────────────────────────────────────────

fsl_bold_nuisance_regression() {
  local input="$1"; local output="$2"; local brain_mask="$3"; local csf_mask="$4"; local wm_mask="$5"
  local nuisance_txt="$6"; local nuisance_mat="$7"; local tr="$8"; local highpass="$9"; local lowpass="${10}"; local mode="${11}"
  require_file "${input}"; require_file "${brain_mask}"; require_file "${nuisance_txt}"; require_mode "${mode}"; make_parent_dir "${output}"

  # Save global mean before regression (fsl_glm --demean zeros the mean).
  # fslstats appends trailing space; trim to avoid fslmaths filename-parsing bug.
  local global_mean
  global_mean=$(fslstats "${input}" -k "${brain_mask}" -m | tr -d ' ')

  # Convert the Python-generated final nuisance matrix for fsl_glm.
  Text2Vest "${nuisance_txt}" "${nuisance_mat}"

  # Write residuals to temp file (fslmaths cannot read+write same file).
  local tmp_res
  tmp_res="$(dirname "${output}")/tmp_nuisance_residuals.nii.gz"
  fsl_glm -i "${input}" -d "${nuisance_mat}" -m "${brain_mask}" \
    --demean --out_res="${tmp_res}"

  # Restore global mean. fslmaths misparses negative floats (e.g. -0.000000).
  fslmaths "${tmp_res}" -add "${global_mean}" "${output}"
  rm -f "${tmp_res}"
}

# ── bold_s09: temporal bandpass filtering ──────────────────────────────────

fsl_bold_temporal_filtering() {
  local input="$1"; local output="$2"; local hp_sigma="$3"; local lp_sigma="$4"; local mode="$5"
  require_file "${input}"; require_mode "${mode}"; make_parent_dir "${output}"

  local mean_before
  mean_before=$(fslstats "${input}" -m | tr -d ' ')

  fslmaths "${input}" -bptf "${hp_sigma}" "${lp_sigma}" -add "${mean_before}" "${output}"
}

# ── bold_s10: grand mean scaling to target ────────────────────────────────

fsl_bold_grand_mean_scaling() {
  local input="$1"; local mask="$2"; local output="$3"; local target="$4"; local mode="$5"
  require_file "${input}"; require_file "${mask}"; require_mode "${mode}"; make_parent_dir "${output}"

  local mean_val
  mean_val=$(fslstats "${input}" -k "${mask}" -m | tr -d ' ')
  fslmaths "${input}" -div "${mean_val}" -mul "${target}" "${output}"
}

# ── bold_s11: EPI → MNI via convertwarp + applywarp ───────────────────────

fsl_bold_space_mni152() {
  local input="$1"; local template="$2"; local epi_mat="$3"; local anat_warp="$4"; local epi_mask="$5"; local output="$6"; local tr="$7"; local mode="$8"
  require_file "${input}"; require_file "${template}"; require_file "${epi_mat}"; require_file "${anat_warp}"; require_file "${epi_mask}"
  require_mode "${mode}"; make_parent_dir "${output}"

  local masked_input
  masked_input="$(dirname "${output}")/tmp_epi_masked_before_mni.nii.gz"
  fslmaths "${input}" -mas "${epi_mask}" "${masked_input}"

  local epi_to_mni_warp
  epi_to_mni_warp="$(dirname "${output}")/tmp_fsl_epi_to_mni_warp.nii.gz"

  convertwarp -r "${template}" -o "${epi_to_mni_warp}" \
    -m "${epi_mat}" -w "${anat_warp}"

  applywarp -i "${masked_input}" -r "${template}" -w "${epi_to_mni_warp}" \
    --interp=spline -o "${output}"

  rm -f "${masked_input}" "${epi_to_mni_warp}"

  # applywarp can leave 4D outputs with bad MNI geometry. Copy the template
  # qform/sform without changing dimensions, then restore only pixdim4.
  fslcpgeom "${template}" "${output}" -d
  python3 - "${output}" "${tr}" <<'PY'
from pathlib import Path
import sys

import nibabel as nib

path = Path(sys.argv[1])
tr = float(sys.argv[2])
img = nib.load(str(path))
header = img.header.copy()
zooms = list(header.get_zooms())
if len(zooms) < 4:
    zooms = zooms[:3] + [tr]
else:
    zooms[3] = tr
header.set_zooms(tuple(zooms))
fixed = nib.Nifti1Image(img.get_fdata(dtype="float32"), img.affine, header)
fixed.set_qform(img.get_qform(), int(img.header["qform_code"]))
fixed.set_sform(img.get_sform(), int(img.header["sform_code"]))
nib.save(fixed, str(path))
PY
}

# ── bold_s12: spatial smoothing ───────────────────────────────────────────

fsl_bold_smooth() {
  local input="$1"; local output="$2"; local sigma="$3"; local mode="$4"
  require_file "${input}"; require_mode "${mode}"; make_parent_dir "${output}"
  fslmaths "${input}" -kernel gauss "${sigma}" -fmean "${output}"
}

# When called directly, dispatch first argument as function name.
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  dispatch "$@"
fi
