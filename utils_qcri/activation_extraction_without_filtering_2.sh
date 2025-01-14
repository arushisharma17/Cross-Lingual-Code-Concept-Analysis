#!/bin/bash -l

conda activate neurox_pip

# Print usage instructions
usage() {
  echo "Usage: $0 [options]"
  echo
  echo "Options:"
  echo "  --inputPath <path>         Path to input sentence files (default: Data/Java-CS)."
  echo "  --model <name>             Model to extract activations for (default: Salesforce/codet5-base)."
  echo "  --layer <int>              Layer to extract activations for (default: 0)."
  echo "  --sentence_length <int>    Maximum sentence length for filtering (default: 300)."
  echo "  --minfreq <int>            Minimum frequency for token filtering (default: 0)."
  echo "  --maxfreq <int>            Maximum frequency for token filtering (default: 15)."
  echo "  --delfreq <int>            Delete frequency threshold for rare tokens (default: 10000000)."
  echo "  --outputDir <path>         Custom output directory (default: derived from model and input path)."
  echo "  -h, --help                 Display this help message and exit."
  echo
  echo "Example:"
  echo "  bash $0 --inputPath Data/Java-CS --model bert-base-uncased --layer 2"
  exit 0
}

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --inputPath)
      inputPath="$2"
      shift
      shift
      ;;
    --model)
      model="$2"
      shift
      shift
      ;;
    --layer)
      layer="$2"
      shift
      shift
      ;;
    --sentence_length)
      sentence_length="$2"
      shift
      shift
      ;;
    --minfreq)
      minfreq="$2"
      shift
      shift
      ;;
    --maxfreq)
      maxfreq="$2"
      shift
      shift
      ;;
    --delfreq)
      delfreq="$2"
      shift
      shift
      ;;
    --outputDir)
      customOutputDir="$2"
      shift
      shift
      ;;
    -h|--help)
      usage
      ;;
    *)
      echo "Unknown option: $1"
      usage
      ;;
  esac
done

# Default values
inputPath="${inputPath:-Data/Java-CS}"
model="${model:-Salesforce/codet5-base}"
layer="${layer:-0}"
sentence_length="${sentence_length:-300}"
minfreq="${minfreq:-0}"
maxfreq="${maxfreq:-15}"
delfreq="${delfreq:-10000000}"
scriptDir="ConceptX/scripts/"
encoder_input="input.in"
decoder_input="label.out"
NEUROX_PATH="NeuroX/scripts"

datasetname=$(echo "$inputPath" | sed 's/[\/]/_/g')
modelname=$(echo "$model" | sed 's/[\/]/_/g')
outputDir="${customOutputDir:-Experiments/${modelname}/${datasetname}/layer${layer}/extraction_without_filtering}"

# Create output directory and save configuration
if [ -d "$outputDir" ]; then
  rm -rf "$outputDir"
fi
mkdir -p "$outputDir"

config_file="${outputDir}/config.txt"
cat << EOF > "$config_file"
Input Path: $inputPath
Model: $model
Layer: $layer
Sentence Length: $sentence_length
Min Frequency: $minfreq
Max Frequency: $maxfreq
Delete Frequency: $delfreq
Output Directory: $outputDir
EOF

# Copy input files
cp "${inputPath}/$encoder_input" "${outputDir}/$encoder_input.tok"
cp "${inputPath}/$decoder_input" "${outputDir}/$decoder_input.tok"

encoder_working_file="${outputDir}/${encoder_input}.tok.sent_len"
decoder_working_file="${outputDir}/${decoder_input}.tok.sent_len"

# Sentence length filtering
python "code/parallel_sentence_length.py" \
    --encoder_input "${outputDir}/$encoder_input.tok" \
    --decoder_input "${outputDir}/$decoder_input.tok" \
    --encoder_output_file "$encoder_working_file" \
    --decoder_output_file "$decoder_working_file" \
    --length "$sentence_length"

# Calculate vocabulary size
python "${scriptDir}/frequency_count.py" \
    --input-file "$encoder_working_file" \
    --output-file "${encoder_working_file}.words_freq"

python "${scriptDir}/frequency_count.py" \
    --input-file "$decoder_working_file" \
    --output-file "${decoder_working_file}.words_freq"

# Extract layer activations
PYTHONPATH="$NEUROX_PATH" python3 -u NeuroX/neurox/data/extraction/transformers_extractor.py \
    "${model}" \
    "$encoder_working_file" "$decoder_working_file" \
    "${outputDir}/activations.json" \
    --output_type json \
    --seq2seq_component both \
    --decompose_layers \
    --filter_layers "$layer"

# Create dataset file
python "${scriptDir}/create_data_single_layer.py" \
    --text-file "$encoder_working_file" \
    --activation-file "${outputDir}/encoder-activations-layer${layer}.json" \
    --output-prefix "$encoder_working_file"

python "${scriptDir}/create_data_single_layer.py" \
    --text-file "$decoder_working_file" \
    --activation-file "${outputDir}/decoder-activations-layer${layer}.json" \
    --output-prefix "$decoder_working_file"

# Frequency filtering
python -u "${scriptDir}/frequency_filter_data.py" \
    --input-file "${encoder_working_file}-dataset.json" \
    --frequency-file "${encoder_working_file}.words_freq" \
    --sentence-file "${encoder_working_file}-sentences.json" \
    --minimum-frequency "$minfreq" \
    --maximum-frequency "$maxfreq" \
    --delete-frequency "$delfreq" \
    --output-file "$encoder_working_file"  > "$outputDir/encoder_ff_log.txt" 2>&1

python -u "${scriptDir}/frequency_filter_data.py" \
    --input-file "${decoder_working_file}-dataset.json" \
    --frequency-file "${decoder_working_file}.words_freq" \
    --sentence-file "${decoder_working_file}-sentences.json" \
    --minimum-frequency "$minfreq" \
    --maximum-frequency "$maxfreq" \
    --delete-frequency "$delfreq" \
    --output-file "$decoder_working_file" > "$outputDir/decoder_ff_log.txt" 2>&1

# Step 5: Extract vectors
python -u "${scriptDir}/extract_data.py" \
    --input-file "${encoder_working_file}_min_${minfreq}_max_${maxfreq}_del_${delfreq}-dataset.json" \
    --output-vocab-file "${outputDir}/encoder-processed-vocab.npy" \
    --output-point-file "${outputDir}/encoder-processed-point.npy"

python -u "${scriptDir}/extract_data.py" \
    --input-file "${decoder_working_file}_min_${minfreq}_max_${maxfreq}_del_${delfreq}-dataset.json" \
    --output-vocab-file "${outputDir}/decoder-processed-vocab.npy" \
    --output-point-file "${outputDir}/decoder-processed-point.npy"

echo "Experiment completed: $modelname, Dataset: $datasetname, Layer: $layer"