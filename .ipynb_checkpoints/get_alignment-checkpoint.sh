#!/bin/bash -l

top_n_translations="2" 
matching_threshold=".5"
size_threshold="1"
types="1"
cluster_file_path1="cluster-output/clusters-kmeans-3.txt"
cluster_file_path2="cluster-output/clusters-kmeans-3-5.txt"
dictionary_file_path="Dictionary/English-German.txt"

python -u "code/alignClusters.py" $cluster_file_path1 $cluster_file_path2 $dictionary_file_path $top_n_translations $matching_threshold $size_threshold $types
