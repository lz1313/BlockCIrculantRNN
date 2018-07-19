#!/bin/bash

target_dir=$1

fnames=(`find $target_dir -name "*.WAV"`)

for fname in "${fnames[@]}"
do
  mv "$fname" "${fname%.wav}.nist"
  sox "${fname%.wav}.nist" -t wav "$fname"
  if [ $? = 0 ]; then
    echo renamed $fname to nist and converted back to wav using sox
  else
    mv "${fname%.wav}.nist" "$fname"
  fi
done
