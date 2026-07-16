#!/usr/bin/env bash

# Load a trusted, shell-compatible .env file without overriding variables that
# were explicitly supplied by the caller (for example CONFIG=... ./script.sh).
flowfactory_load_env() {
  local env_file=$1
  local raw line name source_status=0
  local allexport_was_set=0
  local -a names=()
  local -A seen=()
  local -A caller_values=()

  if [[ ! -f "${env_file}" ]]; then
    printf 'error: environment file not found: %s\n' "${env_file}" >&2
    printf 'copy .env.example to .env and set the machine-specific values\n' >&2
    return 1
  fi

  # Validate the file shape and remember caller-provided values. The file is
  # sourced below so quoted values and references to earlier entries work.
  while IFS= read -r raw || [[ -n "${raw}" ]]; do
    raw=${raw%$'\r'}
    line=${raw#"${raw%%[![:space:]]*}"}
    if [[ -z "${line}" || "${line}" == \#* ]]; then
      continue
    fi
    if [[ "${line}" =~ ^(export[[:space:]]+)?([A-Za-z_][A-Za-z0-9_]*)= ]]; then
      name=${BASH_REMATCH[2]}
    else
      printf 'error: invalid .env entry in %s: %s\n' "${env_file}" "${raw}" >&2
      return 1
    fi
    if [[ -z "${seen[${name}]+x}" ]]; then
      names+=("${name}")
      seen[${name}]=1
      if [[ -v "${name}" ]]; then
        caller_values[${name}]=${!name}
      fi
    fi
  done < "${env_file}"

  [[ $- == *a* ]] && allexport_was_set=1
  set -a
  if source "${env_file}"; then
    source_status=0
  else
    source_status=$?
  fi
  (( allexport_was_set == 1 )) || set +a

  for name in "${names[@]}"; do
    if [[ -n "${caller_values[${name}]+x}" ]]; then
      printf -v "${name}" '%s' "${caller_values[${name}]}"
      export "${name}"
    fi
  done

  if (( source_status != 0 )); then
    printf 'error: failed to load environment file: %s\n' "${env_file}" >&2
    return "${source_status}"
  fi
}

flowfactory_require_env() {
  local name
  for name in "$@"; do
    if [[ -z "${!name:-}" ]]; then
      printf 'error: required environment variable is empty: %s\n' "${name}" >&2
      return 1
    fi
  done
}
