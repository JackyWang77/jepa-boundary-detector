#!/bin/bash
# E2-JEPA end-to-end demo: cache -> 4 arms -> eval -> summary
set -e
cd "$(dirname "$0")"

echo "== [1/3] build data cache (decode teleop mp4 once) =="
python3 -c "from e2jepa.data import build_cache; build_cache()"

echo "== [2/3] train four arms =="
for arm in e2jepa jepa energy_only bc; do
  echo "-- arm: $arm"
  python3 -m e2jepa.train --arm $arm --wm_steps 3000 --bc_steps 20000
done

echo "== [3/3] evaluate =="
python3 -m e2jepa.eval --episodes 25 --max_steps 300

echo "done. see results/"
