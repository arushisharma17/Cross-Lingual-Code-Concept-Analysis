#!/bin/bash -l

# Now activate the conda environment
conda activate neurox_pip

# Print usage instructions
usage() {
  echo "Usage: $0 [options]"
  echo
  echo "Options:"
  echo "  --inputPath <path>         Path to input sentence files (default: Data/Java-CS)."
  echo "  --model <name>             Model to extract activations for (default: Salesforce/codet5-base)."
  # echo "  --model_class <name>       Model class for activation extraction (default: T5ForConditionalGeneration)."
  echo "  --layer <int>              Layer to extract activations for (default: 0)."
  echo "  --sentence_length <int>    Maximum sentence length for filtering (default: 1000)."
  echo "  --minfreq <int>            Minimum frequency for token filtering (default: 0)."
  echo "  --maxfreq <int>            Maximum frequency for token filtering (default: 15)."
  echo "  --delfreq <int>            Delete frequency threshold for rare tokens (default: 10000000)."
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
    --model_class)
      model_class="$2"
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
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Default values for arguments
inputPath="${inputPath:-Data/Java-CS}"  # Default input path
model="${model:-Salesforce/codet5-base}"  # Default model
model_class="${model_class:-T5ForConditionalGeneration}"  # Default model class
layer="${layer:-0}"  # Default layer
sentence_length="${sentence_length:-5000}"  # Default sentence length
minfreq="${minfreq:-0}"  # Default min frequency
maxfreq="${maxfreq:-15}"  # Default max frequency
delfreq="${delfreq:-10000000}"  # Default delete frequency

# Dynamically derive dataset name
datasetname=$(echo "$inputPath" | sed 's/[\/]/_/g')  # Replace "/" with "_"
modelname=$(echo "$model" | sed 's/[\/]/_/g')

# Specify paths
scriptDir="ConceptX/scripts/"
encoder_input="input.in"           # Encoder sentences
decoder_input="label.out"          # Decoder sentences
NEUROX_PATH="NeuroX/scripts/"      # Path to the modified NeuroX library
mapping="${inputPath}/mapping_dict.json"  # FastAlign mapping for filtering

# Output directory structure
base_output_dir="Experiments"      # Base directory for experiments
outputDir="${base_output_dir}/${modelname}/${datasetname}/layer${layer}/extraction_with_filtering"

# Create output directory and record configuration
if [ -d "$outputDir" ]; then
    rm -rf "$outputDir"
fi
mkdir -p "$outputDir"

# Save configuration to a file
config_file="${outputDir}/config.txt"
cat << EOF > $config_file
Input Path: $inputPath
Model: $model
Dataset Name: $datasetname
Layer: $layer
Sentence Length: $sentence_length
Min Frequency: $minfreq
Max Frequency: $maxfreq
Delete Frequency: $delfreq
Encoder Input: $encoder_input
Decoder Input: $decoder_input
Output Directory: $outputDir
EOF

# Copy input files to the output directory
cp "${inputPath}/$encoder_input" "${outputDir}/$encoder_input.tok"
cp "${inputPath}/$decoder_input" "${outputDir}/$decoder_input.tok"

encoder_working_file="${outputDir}/${encoder_input}.tok.sent_len"
decoder_working_file="${outputDir}/${decoder_input}.tok.sent_len"

# Step 1: Sentence length filtering
python "code/parallel_sentence_length.py" \
    --encoder_input "${outputDir}/$encoder_input.tok" \
    --decoder_input "${outputDir}/$decoder_input.tok" \
    --encoder_output_file "$encoder_working_file" \
    --decoder_output_file "$decoder_working_file" \
    --length "$sentence_length"

# Step 2: Extract activations NOTE: first arg was originally "${model},${model},{model_class}"
PYTHONPATH="$NEUROX_PATH" python -u NeuroX/neurox/data/extraction/transformers_extractor.py \
    "${model}" \
    "$encoder_working_file" "$decoder_working_file" \
    "${outputDir}/activations.json" \
    --output_type json \
    --seq2seq_component both \
    --decompose_layers \
    --filter_layers "$layer"
    
# Step 3: Create dataset files
python "${scriptDir}/create_data_single_layer.py" \
    --text-file "$encoder_working_file" \
    --activation-file "${outputDir}/encoder-activations-layer${layer}.json" \
    --output-prefix "$encoder_working_file"

python "${scriptDir}/create_data_single_layer.py" \
    --text-file "$decoder_working_file" \
    --activation-file "${outputDir}/decoder-activations-layer${layer}.json" \
    --output-prefix "$decoder_working_file"

# Step 4: Frequency filtering
python -u "code/parallel_frequency_filter_data.py" \
    --src-dataset "${encoder_working_file}-dataset.json" \
    --src-sentences "${encoder_working_file}-sentences.json" \
    --tgt-dataset "${decoder_working_file}-dataset.json" \
    --tgt-sentences "${decoder_working_file}-sentences.json" \
    --mapping-dict "$mapping" \
    --output-src-file-prefix "$encoder_working_file" \
    --output-tgt-file-prefix "$decoder_working_file" \
    --minimum-frequency "$minfreq" \
    --maximum-frequency "$maxfreq" \
    --delete-frequency "$delfreq"

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
echo "Saving files:"
echo "${outputDir}/encoder-processed-vocab.npy"
echo "${outputDir}/encoder-processed-point.npy"
echo "${outputDir}/decoder-processed-vocab.npy"
echo "${outputDir}/decoder-processed-point.npy"

