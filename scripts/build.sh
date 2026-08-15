#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

rm -rf dist
npx tsc

cp -r src/weights dist/

# Data is optional: `npm run data` fetches it; the XOR example works without it.
if [ -d src/data ]; then
    cp -r src/data dist/
fi
