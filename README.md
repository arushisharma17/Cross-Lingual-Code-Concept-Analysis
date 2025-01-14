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
./utils_qcri/clustering_2.sh --inputPath Experiments/Salesforce_codet5-base/Data_CPP-Cuda/layer0/extraction_without_filtering --clusters 500 --mode visualize
```

### Aligning Clusters

```bash
#dos2unix utils_qcri/get_alignment_2.sh
utils_qcri/get_alignment_2.sh --clusterDir Experiments/Salesforce_codet5-base/Data_CPP-Cuda/layer0/extraction_without_filtering/clustering --dictionary Data/CPP-Cuda/dictionary.json
```

Note: Change the inputPath to the corpus you want to run clustering for like Experiments/google-t5_t5-base/Data_Java-CS/layer3/extraction_without_filtering
