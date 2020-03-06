#!/usr/bin/env bash

SRC_DIR="$(cd "$(dirname ${BASH_SOURCE[0]})" && pwd)"
if [[ "$(hostname)" == login* ]]; then
    source "${SRC_DIR}/seawulf_modules.txt"
fi
source activate ${SRC_DIR}/venv
