# Cross-Lingual-Code-Concept-Analysis

Write up forthcoming. In the mean time -- read `Full Demo.ipynb` for details. 

# Running the code on WSL

## Setup and environment 

```bash
conda env create -f environment.yml
conda activate neurox_pip
pip install --upgrade torch==2.0.0
pip install --upgrade transformers==4.30.0
```

Install Rust:

Download the latest version of Rust: https://rustup.rs/

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

Clone ConceptX:

```bash
git clone https://github.com/hsajjad/ConceptX
```

## Run Through for CPP-Cuda

This is the run through for the CPP-Cuda corpus.
To run through for other corpora, change the corpus name in the following steps.

### Tokenizing the corpus

```bash
python -u tree-sitter/tokenize_corpus.py Data/CPP-Cuda/cpp-cuda.txt cpp cuda --level leaf
```
Note: Change the Data/CPP-Cuda/cpp-cuda.txt to the corpus you want to tokenize like Data/Java-CS/java-cs.txt

### Collecting Alignments

#### Using awesome-align

```bash
awesome-align \
  --model_name_or_path bert-base-multilingual-cased \
  --data_file Data/CPP-Cuda/tree_sitter_tokenized_leaf.txt \
  --output_file Data/CPP-Cuda/forward.align \
  --extraction softmax \
  --batch_size 32
```

Note: Change the Data/CPP-Cuda/cpp-cuda.txt to the corpus you want to make alignment like Data/Java-CS/java-cs.txt

### Create Translation Dictionary

```bash
python -u utils/wordlevel_dict_text.py Data/CPP-Cuda/cpp-cuda.txt Data/CPP-Cuda/forward.align
python -u utils/wordlevel_dict.py Data/CPP-Cuda/cpp-cuda.txt Data/CPP-Cuda/forward.align
```

Note: Change the Data/CPP-Cuda/cpp-cuda.txt to the corpus you want to make translation dictionary like Data/Java-CS/java-cs.txt

### Making Mapping Dictionary

Using awesome-align

```bash
python -u utils/mapping_awesomealign.py Data/CPP-Cuda/cpp-cuda.txt Data/CPP-Cuda/forward.align
```
Note: Change the Data/CPP-Cuda/cpp-cuda.txt to the corpus you want to make mapping dictionary like Data/Java-CS/java-cs.txt

### Splitting the corpus into Encoder Decoder Pieces

```bash
python -u utils/split.py Data/CPP-Cuda/cpp-cuda.txt
```

Note: Change the Data/CPP-Cuda/cpp-cuda.txt to the corpus you want to split into encoder decoder pieces like Data/Java-CS/java-cs.txt

#### Running Activation Extraction 

```bash
mkdir -p cache
export HF_HOME="./cache/"
```

##### Concept Alignment Experiment

```bash
#dos2unix utils_qcri/activation_extraction_with_filtering_2.sh
./utils_qcri/activation_extraction_with_filtering_2.sh --model Salesforce/codet5-base  --inputPath Data/CPP-Cuda/ --layer 0 --sentence_length 2048 --minfreq 0 --maxfreq 1500000 --delfreq 10000000
```

Note: Change the inputPath to the corpus you want to run activation extraction for overlap experiment like Data/Java-CS/

##### Overlap Experiment

```bash
#dos2unix utils_qcri/activation_extraction_without_filtering_2.sh
./utils_qcri/activation_extraction_without_filtering_2.sh --model Salesforce/codet5-base  --inputPath Data/CPP-Cuda --layer 0 --sentence_length 2048 
```

Note: Change the inputPath to the corpus you want to run activation extraction for overlap experiment like Data/Java-CS/

### Clustering Representations

```bash
#dos2unix utils_qcri/clustering_2.sh
./utils_qcri/clustering_2.sh --inputPath Experiments/Salesforce_codet5-base/Data_CPP-Cuda/extraction_without_filtering --layer 0 --clusters 500 --mode visualize
```
Note: Change the inputPath to the corpus you want to run clustering for like Experiments/google-t5_t5-base/Data_Java-CS/layer3/extraction_without_filtering

Side note: The input path doesnt show the layerdir. We add it in the code and hence need the layer number to be passed.

### Aligning Clusters

```bash
#dos2unix utils_qcri/get_alignment_2.sh
utils_qcri/get_alignment_2.sh --clusterDir Experiments/Salesforce_codet5-base/Data_CPP-Cuda/layer0/extraction_without_filtering/clustering --dictionary Data/CPP-Cuda/dictionary.json
```

Note: Change the inputPath to the corpus you want to run clustering for like Experiments/google-t5_t5-base/Data_Java-CS/layer3/extraction_without_filtering

Side note: The input path doesnt show the layerdir. We add it in the code and hence need the layer number to be passed.

### Labelling Clusters

Exit wsl for installing google-generativeai as it is not available in python3.8 which is the version of python in the conda environment.

```bash
conda deactivate neurox_pip
exit
```
Add a .env file in the LLM_labelling folder with the following content:
```
GEMINI_API_KEY=your_gemini_api_key
```
Note: Use paid gemini api key. If not then chnage the code to pause after 15 clusters and run it in loop.

Install google-generativeai:

```bash
python -m venv venv
source venv/bin/activate # or venv\Scripts\activate on Windows
pip install google-generativeai==0.8.3
```

```bash
python LLM_labelling/gemini_labelling_cpp.py --sentence-file Data/CPP-Cuda/input.in  --model-dir Experiments/Salesforce_codet5-base/Data_CPP-Cuda --dir-extension extraction_without_filtering/clustering --component encoder --start-layer 0 --end-layer 12
```

```bash
python LLM_labelling/gemini_labelling_cuda.py --sentence-file Data/CPP-Cuda/label.out  --model-dir Experiments/Salesforce_codet5-base/Data_CPP-Cuda --dir-extension extraction_without_filtering/clustering --component decoder --start-layer 0 --end-layer 12
```

Note: Change the input.in to the corpus you want to label like Data/Java-CS/input.in, change python script to the language you want to label like gemini_labelling_cpp.py, gemini_labelling_cuda.py, gemini_labelling_java.py, gemini_labelling_csharp.py

### Combining Alignments and LLM Labels

```bash
python code/combine_alignments_and_llm_labels.py --model-dir Experiments/Salesforce_codet5-base/Data_CPP-Cuda --dir-extension extraction_without_filtering/clustering --start-layer 0 --end-layer 12
```

Note: Change the base-dir to the directory you want to combine alignments and LLM labels for like Experiments/Salesforce_codet5-base/Data_CPP-Cuda

### Top Semantic Tags

Finds the top 20 semantic tags across all layers. So it is a good idea to run this after LLM labels for all layers.

```bash
python code/topSemantic.py --model-dir Experiments/Salesforce_codet5-base/Data_CPP-Cuda --dir-extension extraction_without_filtering/clustering
```

Note: Change the model-dir to the directory you want to analyze semantic tags for like Experiments/Salesforce_codet5-base/Data_CPP-Cuda

### Visualizing Clusters

```bash
source venv/bin/activate # or venv\Scripts\activate on Windows
pip install -r Visualize/requirements.txt
```

```bash
streamlit run Visualize/app.py --model-dir Experiments/Salesforce_codet5-base/Data_CPP-Cuda --dir-extension extraction_without_filtering/clustering --encoder-file Data/CPP-Cuda/input.in --decoder-file Data/CPP-Cuda/label.out
```

Note: Change the model-dir to the directory you want to visualize clusters for like Experiments/Salesforce_codet5-base/Data_CPP-Cuda. Chnage the sentence-file to the corpus you want to visualize like Data/CPP-Cuda/input.in, Data/CPP-Cuda/label.out

