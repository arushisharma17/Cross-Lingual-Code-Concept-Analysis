#!/bin/bash -l

top_n_translations="2" # 5-20 best translations
matching_threshold=".9" # we consider Cs (a concept in language s) to be aligned to Ct a concept in language t) if 60% of its types have a semantically equivalentword in Ct, i.e. θA = 0.8 and 0.7-0.9 range 
size_threshold=".4" # Finally, we also only align concepts Cs/Ct if their sizes do not differ by more than 40%, to avoid aligning very small concepts in one language to a single large concept in another language.

types="1" # Minimal distinct word size

cluster_file_path1="visualize_$1/layer$2/encoder-clusters-kmeans-$1.txt"
cluster_file_path2="visualize_$1/layer$2/decoder-clusters-kmeans-$1.txt"
# dictionary_file_path="Dictionary/en-fr.json"
dictionary_file_path="Data/Java-CS/dictionary.json"

python -u "code/alignClusters.py" $cluster_file_path1 $cluster_file_path2 $dictionary_file_path $top_n_translations $matching_threshold $size_threshold $types
