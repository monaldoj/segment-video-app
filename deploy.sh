#!/usr/bin/env bash
set -euo pipefail

# Deploy with Databricks Asset Bundles (DAB).
# Requires: Databricks CLI configured (auth/host).

databricks bundle deploy -t dev


