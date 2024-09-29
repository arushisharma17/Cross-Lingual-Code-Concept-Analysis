#!/bin/bash -l

conda activate neurox_pip

scriptDir="ConceptX/scripts/"   # path to ConceptX script directory
inputPath="Data/English-German/"       # path to the directory where sentence files are saved
encoder_input="input.in"       # Encoder Sentences
decoder_input="label.out"      # Decoder Sentences
outputDir="activationsOutput/"    # Specify the output directory for the extractions
mkdir -p $outputDir               # Create the output directory if it doesn't exist

# model="bert-base-uncased"     # Model that we want to do the extraction for
# model="google/mt5-base"     # Model that we want to do the extraction for
model="Salesforce/codet5-base"     # Model that we want to do the extraction for
model_class="T5ForConditionalGeneration"
NEUROX_PATH="NeuroX/scripts"

sentence_length=300   # maximum sentence length
minfreq=0 
maxfreq=15
delfreq=10000000 
layers_of_interest="0,1,3,6,9,12" # define layers of interest 
layers_of_interest="9" # define layers of interest 


#specify according to where you want your activations to be saved. Keep a clear structure

# Output files in the new folder
encoder_working_file="${outputDir}/${encoder_input}.tok.sent_len"
decoder_working_file="${outputDir}/${decoder_input}.tok.sent_len"

# Copy input files to the output directory and process them
#cp ${inputPath}/${encoder_input} ${encoder_tok_file}
#cp ${inputPath}/${decoder_input} ${decoder_tok_file}

cp ${inputPath}/$encoder_input ${outputDir}/$encoder_input.tok
cp ${inputPath}/$decoder_input ${outputDir}/$decoder_input.tok


# Do sentence length filtering and keep sentences max length of {sentence_length}
python "code/parallel_sentence_length.py" --encoder_input ${outputDir}/$encoder_input.tok --decoder_input ${outputDir}/$decoder_input.tok --encoder_output_file $encoder_working_file --decoder_output_file  $decoder_working_file --length ${sentence_length}


# Calculate vocabulary size
python ${scriptDir}/frequency_count.py --input-file ${encoder_working_file} --output-file ${encoder_working_file}.words_freq
python ${scriptDir}/frequency_count.py --input-file ${decoder_working_file} --output-file ${decoder_working_file}.words_freq


# Extract layer-wise activations
PYTHONPATH=$NEUROX_PATH python3 -u NeuroX/neurox/data/extraction/transformers_extractor.py "${model},${model},${model_class}" ${encoder_working_file}  ${decoder_working_file} "activations.json"  --output_type json --seq2seq_component both --decompose_layers --filter_layers "$layers_of_interest"
# PYTHONPATH=$NEUROX_PATH python3 -u NeuroX/neurox/data/extraction/transformers_extractor.py "${model}" ${encoder_working_file}  ${decoder_working_file} activations.json --output_type json --seq2seq_component both --decompose_layers --filter_layers "$layers_of_interest"

# Create a dataset file with word and sentence indexes
layers_of_interest="0 1 3 6 9 12"

layers_of_interest="9"

# Process each layer of interest
for j in $layers_of_interest
do
    python ${scriptDir}/create_data_single_layer.py --text-file ${encoder_working_file} --activation-file encoder-activations-layer${j}.json --output-prefix "${encoder_working_file}_${j}" 
    python ${scriptDir}/create_data_single_layer.py --text-file ${decoder_working_file} --activation-file decoder-activations-layer${j}.json --output-prefix "${decoder_working_file}_${j}"
    
    python ${scriptDir}/frequency_filter_data.py --input-file ${encoder_working_file}_${j}-dataset.json --frequency-file ${encoder_working_file}.words_freq --sentence-file "${encoder_working_file}_${j}-sentences.json" --minimum-frequency $minfreq --maximum-frequency $maxfreq --delete-frequency ${delfreq} --output-file "${encoder_working_file}_${j}"

    python ${scriptDir}/frequency_filter_data.py --input-file ${decoder_working_file}_${j}-dataset.json --frequency-file ${decoder_working_file}.words_freq --sentence-file "${decoder_working_file}_${j}-sentences.json" --minimum-frequency $minfreq --maximum-frequency $maxfreq --delete-frequency ${delfreq} --output-file "${decoder_working_file}_${j}"
    
    python -u ${scriptDir}/extract_data.py --input-file ${encoder_working_file}_${j}_min_${minfreq}_max_${maxfreq}_del_${delfreq}-dataset.json --output-vocab-file ${outputDir}/encoder-processed-vocab-${j}.npy --output-point-file ${outputDir}/encoder-processed-point-${j}.npy
    
    python -u ${scriptDir}/extract_data.py --input-file ${decoder_working_file}_${j}_min_${minfreq}_max_${maxfreq}_del_${delfreq}-dataset.json --output-vocab-file ${outputDir}/decoder-processed-vocab-${j}.npy --output-point-file ${outputDir}/decoder-processed-point-${j}.npy

done


rm -r ${outputDir}/*-dataset.json
rm -r ${outputDir}/*-labels.json
rm -r *activations*.json
