#!/usr/bin/env bash

set -euo pipefail

require_api_key() {
  if [[ -z "${INFINI_API_KEY:-}" ]]; then
    echo "INFINI_API_KEY is required" >&2
    exit 1
  fi
}

base_url() {
  printf '%s' "${INFINI_BASE_URL:-https://cloud.infini-ai.com}"
}

api_curl() {
  require_api_key
  local endpoint="$1"
  local data="${2:-}"

  if [[ -n "${data}" ]]; then
    curl --request POST \
      --url "$(base_url)${endpoint}" \
      --header 'Accept: application/json' \
      --header "Authorization: Bearer ${INFINI_API_KEY}" \
      --header 'Content-Type: application/json' \
      --data "${data}"
    return
  fi

  curl --request POST \
    --url "$(base_url)${endpoint}" \
    --header 'Accept: application/json' \
    --header "Authorization: Bearer ${INFINI_API_KEY}" \
    --header 'Content-Type: application/json'
}
