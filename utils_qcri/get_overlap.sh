#!/bin/bash -l

clusters_path="cluster-output/encoder-clusters-kmeans-1500.txt"
output_path="overlap.txt"
clusters_threshold=".4" # We deem a concept C to be multilingual or overlapping if all languages being considered form at least 30% (θO = 0.3) of the concept.
sentences_threshold="232" # Threshold at which the sentences are split into two different languages
unique_tokens="3" #Minimal number of tokens should a cluster have

python -u "code/get_overlapping_clusters.py" --cluster_file $clusters_path --output_path $output_path --clusters_threshold $clusters_threshold --sentences_threshold $sentences_threshold --unique_tokens $unique_tokens
