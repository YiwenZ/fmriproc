#!/usr/bin/env bash
set -euo pipefail

require_file() {
  local file="$1"
  if [[ ! -f "${file}" ]]; then
    echo "Required file not found: ${file}" >&2
    exit 1
  fi
}

require_dir() {
  local dir="$1"
  if [[ ! -d "${dir}" ]]; then
    echo "Required directory not found: ${dir}" >&2
    exit 1
  fi
}

require_mode() {
  local mode="$1"
  if [[ "${mode}" != "tested" && "${mode}" != "basic" ]]; then
    echo "Mode must be tested or basic; got ${mode}" >&2
    exit 1
  fi
}

make_parent_dir() {
  local file="$1"
  mkdir -p "$(dirname "${file}")"
}

dispatch() {
  local func_name="$1"
  if [[ -z "${func_name}" ]]; then
    echo "Usage: $0 <function_name> [args...]" >&2
    exit 2
  fi

  if ! declare -f "${func_name}" > /dev/null; then
    echo "Unknown function: ${func_name}" >&2
    exit 3
  fi

  shift
  "${func_name}" "$@"
}

# When sourced with function name as first argument, dispatch.
# This supports both direct sourcing and `bash script.sh func_name args...` usage.
if [[ "${BASH_SOURCE[0]}" != "${0}" ]]; then
  # Being sourced by another script; do nothing (functions are available).
  :
fi
