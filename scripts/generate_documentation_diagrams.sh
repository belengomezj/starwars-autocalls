#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE_DIR="$ROOT_DIR/docs/assets/diagrams"
OUTPUT_DIR="$ROOT_DIR/docs/assets/figures"
MERMAID_CLI_VERSION="11.12.0"

for diagram in architecture_end_to_end architecture_services preprocessing_point_in_time; do
  npx --yes "@mermaid-js/mermaid-cli@$MERMAID_CLI_VERSION" \
    --input "$SOURCE_DIR/$diagram.mmd" \
    --output "$OUTPUT_DIR/$diagram.png" \
    --backgroundColor white \
    --scale 2
done
