#! /usr/bin/env bash
# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
if [ "$#" -eq 4 ]; then
   echo "You are running the text script with:"
   echo "type:$1"
   echo "num_examples:$2"
   echo "vocab_size:$3"
   echo "max_len:$4"
   SPECIAL_LEN=5
elif [ "$#" -eq 5 ]; then
   echo "You are running the text script with:"
   echo "type:$1"
   echo "num_examples:$2"
   echo "vocab_size:$3"
   echo "max_len:$4"
   echo "special:$5"
   SPECIAL_LEN=${SPECIAL_LEN:-$5}
else
   echo "toy.sh DATA_TYPE NUMEXAMPLES VOCAB_SIZE MAX_LEN"
   exit 1
fi
set -e

BASE_DIR="."

DATA_TYPE=${DATA_TYPE:-$1}
NUMEXAMPLES=${NUMEXAMPLES:-$2}
VOCAB_SIZE=${VOCAB_SIZE:-$3}
MAX_LEN=${MAX_LEN:-$4}


echo "Using type=${DATA_TYPE}. To change this set DATA_TYPE to 'copy' or 'reverse' or 'sort'"

OUTPUT_DIR=${OUTPUT_DIR:-./toy_data/toy_${DATA_TYPE}}
echo "Writing to ${OUTPUT_DIR}. To change this, set the OUTPUT_DIR environment variable."

OUTPUT_DIR_TRAIN="${OUTPUT_DIR}/train"
OUTPUT_DIR_DEV="${OUTPUT_DIR}/dev"
OUTPUT_DIR_TEST="${OUTPUT_DIR}/test"

mkdir -p $OUTPUT_DIR

# Write train, dev and test data
${BASE_DIR}/generate_toy_data.py  \
  --type ${DATA_TYPE} \
  --num_examples ${NUMEXAMPLES} \
  --vocab_size ${VOCAB_SIZE} \
  --max_len ${MAX_LEN} \
  --special ${SPECIAL_LEN} \
  --output_dir ${OUTPUT_DIR_TRAIN}

${BASE_DIR}/generate_toy_data.py  \
  --type ${DATA_TYPE} \
  --num_examples ${NUMEXAMPLES} \
  --vocab_size ${VOCAB_SIZE} \
  --max_len ${MAX_LEN} \
  --special ${SPECIAL_LEN} \
  --output_dir ${OUTPUT_DIR_DEV}

${BASE_DIR}/generate_toy_data.py  \
  --type ${DATA_TYPE} \
  --num_examples ${NUMEXAMPLES} \
  --vocab_size ${VOCAB_SIZE} \
  --max_len ${MAX_LEN} \
  --special ${SPECIAL_LEN} \
  --output_dir ${OUTPUT_DIR_TEST}

# Create Vocabulary
${BASE_DIR}/generate_vocab.py \
  < ${OUTPUT_DIR_TRAIN}/sources.txt \
  > ${OUTPUT_DIR_TRAIN}/vocab.sources.txt
echo "Wrote ${OUTPUT_DIR_TRAIN}/vocab.sources.txt"

${BASE_DIR}/generate_vocab.py \
  < ${OUTPUT_DIR_TRAIN}/targets.txt \
  > ${OUTPUT_DIR_TRAIN}/vocab.targets.txt
echo "Wrote ${OUTPUT_DIR_TRAIN}/vocab.targets.txt"

# Optionally encode data with google/sentencepice
# Useful for testing
if [ "$SENTENCEPIECE" = true ]; then
  spm_train \
    --input=${OUTPUT_DIR_TRAIN}/sources.txt,${OUTPUT_DIR_TRAIN}/targets.txt \
    --model_prefix=${OUTPUT_DIR}/bpe \
    --vocab_size=20 \
    --model_type=bpe
  for dir in ${OUTPUT_DIR_TRAIN} ${OUTPUT_DIR_DEV} ${OUTPUT_DIR_TEST}; do
    spm_encode --model=${OUTPUT_DIR}/bpe.model --output_format=piece \
      < ${dir}/sources.txt \
      > ${dir}/sources.bpe.txt
    spm_encode --model=${OUTPUT_DIR}/bpe.model --output_format=piece \
      < ${dir}/targets.txt \
      > ${dir}/targets.bpe.txt
  done
fi


