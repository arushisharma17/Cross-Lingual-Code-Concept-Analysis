#!/bin/bash -l

conda activate neurox_pip

scriptDir="ConceptX/scripts/"   # path to ConceptX script directory
inputPath="Data/Java-CS/"       # path to the directory where sentence files are saved
# inputPath="Data/EN-FR/"       # path to the directory where sentence files are saved
encoder_input="input.in"       # Encoder Sentences
decoder_input="label.out"      # Decoder Sentences
outputDir="activationsOutput_filtering/"    # Specify the output directory for the extractions

# Delete the output directory if it exists
if [ -d "$outputDir" ]; then
    rm -rf "$outputDir"
fi

mkdir -p $outputDir               # Create the output directory if it doesn't exist

model="Salesforce/codet5-base"     # Model that we want to do the extraction for
model_class="T5ForConditionalGeneration"
NEUROX_PATH="NeuroX/scripts/"

# Filtering parameters; set the layer according to the layer that you want to extract for
sentence_length=1000
minfreq=0
maxfreq=15
delfreq=10000000 

layer=0

# Define the mapping for the filetering
mapping="Data/Java-CS/dictionary.json"

encoder_working_file="${outputDir}/${encoder_input}.tok.sent_len"
decoder_working_file="${outputDir}/${decoder_input}.tok.sent_len"

cp ${inputPath}/$encoder_input ${outputDir}/$encoder_input.tok
cp ${inputPath}/$decoder_input ${outputDir}/$decoder_input.tok

# Do sentence length filtering and keep sentences max length of {sentence_length}
python "code/parallel_sentence_length.py" --encoder_input ${outputDir}/$encoder_input.tok --decoder_input ${outputDir}/$decoder_input.tok --encoder_output_file $encoder_working_file --decoder_output_file  $decoder_working_file --length ${sentence_length}

PYTHONPATH=$NEUROX_PATH python -u NeuroX/neurox/data/extraction/transformers_extractor.py "${model},${model},${model_class}" ${encoder_working_file}  ${decoder_working_file} activations.json --output_type json --seq2seq_component both --decompose_layers --filter_layers ${layer} --device ${device}

# Create a dataset file with word and sentence indexes
python ${scriptDir}/create_data_single_layer.py --text-file ${encoder_working_file} --activation-file encoder-activations-layer${layer}.json --output-prefix ${encoder_working_file} 

python ${scriptDir}/create_data_single_layer.py --text-file ${decoder_working_file} --activation-file decoder-activations-layer${layer}.json --output-prefix ${decoder_working_file}

# Filter number of tokens to fit in the memory for clustering. Input file will be from step 4
python -u "code/parallel_frequency_filter_data.py" --src-dataset ${encoder_working_file}-dataset.json --src-sentences ${encoder_working_file}-sentences.json --tgt-dataset ${decoder_working_file}-dataset.json --tgt-sentences ${decoder_working_file}-sentences.json --mapping-dict $mapping --output-src-file-prefix ${encoder_working_file}  --output-tgt-file-prefix ${decoder_working_file} --minimum-frequency ${minfreq} --maximum-frequency ${maxfreq} --delete-frequency ${delfreq}


# Extract vectors
python -u ${scriptDir}/extract_data.py --input-file ${encoder_working_file}_min_${minfreq}_max_${maxfreq}_del_${delfreq}-dataset.json --output-vocab-file ${outputDir}/encoder-processed-vocab-filtered.npy --output-point-file ${outputDir}/encoder-processed-point-filtered.npy

python -u ${scriptDir}/extract_data.py --input-file ${decoder_working_file}_min_${minfreq}_max_${maxfreq}_del_${delfreq}-dataset.json --output-vocab-file ${outputDir}/decoder-processed-vocab-filtered.npy --output-point-file ${outputDir}/decoder-processed-point-filtered.npy

# rm -r ${outputDir}/*-dataset.json
# rm -r ${outputDir}/*-labels.json
# rm -r *activations*.json
