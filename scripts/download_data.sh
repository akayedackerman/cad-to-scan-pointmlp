#!/usr/bin/env bash
# Download ModelNet40 and ModelNet10 into data/raw/.
# ScanObjectNN has to be requested from its authors (see the end of this script).
#
#   bash scripts/download_data.sh
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/raw && cd data/raw

get() {  # get <url> <folder>
  if [[ -d "$2" ]]; then
    echo "$2 already present"
    return
  fi
  echo "downloading $2 ..."
  curl -fL --retry 3 --progress-bar -o "$2.zip" "$1"
  unzip -q "$2.zip" -d "_$2" && rm "$2.zip"
  mv "_$2/$2" "$2" && rm -rf "_$2"
  echo "$2: $(find "$2" -name '*.off' | wc -l) meshes"
}

get https://modelnet.cs.princeton.edu/ModelNet40.zip ModelNet40
get https://3dvision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip ModelNet10

cat <<'EOF'

ScanObjectNN (needed for the test-time adaptation experiments) is distributed by its authors after
agreeing to their terms of use: https://hkust-vgd.github.io/scanobjectnn/
Download h5_files.zip and extract it so that these files exist:
  data/raw/ScanObjectNN/h5_files/main_split_nobg/test_objectdataset.h5
  data/raw/ScanObjectNN/h5_files/main_split/test_objectdataset.h5
  data/raw/ScanObjectNN/h5_files/main_split/test_objectdataset_augmentedrot_scale75.h5

Optional (one figure only): Stanford Armadillo and Bunny as Z-up OFF meshes in data/raw/stanford/.
EOF
