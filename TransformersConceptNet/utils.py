import random
import sys
import json
import os
import argparse

from collections import Counter, defaultdict
from typing import Dict, List

def cluster_read(fname):
    """
    Given a txt file containing the latent concepts of a corresponding layer, The
    function loads all data in the file and returns it in form of lists. These lists 
    will be then used to create mappings between clusters, words, and sentences 

    Parameters
    ----------
    fname : str
        Path to where the latent concepts data is stored for a corresponding
        layer (Usually saved in a .txt file).

    Returns
    -------
    words: List
        A list of words corresponding to latent concepts of the passed data.
        Each word will be associated with a latent concept (also called a cluster)
    words_idx: List
        A list of word indices corresponding to occurence location of each word
        in the sentences.
    cluster_idx: List
        A list of cluster ids corresponding to the data passed. Each layer will
        have a group of clusters and each cluster contains a group of words
    sent_idx: List
        A list of sentence ids corresponding to which sentences the concept appears in
    """
    words = []
    words_idx = []
    cluster_idx = []
    sent_idx = []
    with open(fname) as f:
        for line in f:
            line  = line.rstrip('\r\n')
            parts = line.split("|||")
            words.append(parts[0])
            cluster_idx.append(int(parts[4]))
            words_idx.append(int(parts[3]))
            sent_idx.append(int(parts[2]))
    return words, words_idx, sent_idx, cluster_idx

def read_cluster_data(fname):
    """
    Given a .txt file corresponding to latent concepts of a layer, the function
    returns a mapping between cluster ids and words. The words corresponding to
    each cluster are returned in a list

    Parameters
    ----------
    fname : str
        Path to where the latent concepts data is stored for a corresponding
        layer (Usually saved in a .txt file).

    Returns
    -------
    clusterToWords: Dict
        A mapping (or dictionary) between cluster and words. The keys of the dictionary
        are clusters, and the corresponding values are words corresponding to that cluster.  
    """
    clusterToWords  = defaultdict(list)
    
    words, words_idx, sent_idx, cluster_idx = cluster_read(fname)
    for i, elem in enumerate(cluster_idx):
        cluster = "c" + str(cluster_idx[i])
        clusterToWords[cluster].append(words[i])
        
    return clusterToWords


def read_annotations(path, num_clusters):
    """
    Given a path to the annotations file; the function 
    returns the LLM annotations for the clusters in the form of 
    a dictionary
    
    Parameters
    ----------
    fname : str
        Path to JSON annotations file
    Returns
    -------
    labels: Dict
        LLM labels for the clusters. 
    """
    
    # Check if annotations file exists
    if not path.exists():
        # If the file doesn't exist, create it and populate it with default cluster names
        default_annotations = {f"c{i}": f"cluster {i}" for i in range(num_clusters)}

        
        # Write default annotations to the file
        with open(str(path), 'w') as file:
            json.dump(default_annotations, file, indent=4)

    
    with open(str(path), "r") as reader:
        labels = json.load(reader)
    
    return labels

    
def read_sentences(path_to_sentences: str) -> List: 
    """
    Given a path to the sentences file, the function returns
    a list of sentences

    Parameters
    ----------
    path_to_sentences: str :
        A path to where the sentences file is stored

    Returns
    -------
    sentences: List
        A list of sentences 
    """
    sentences = []
    with open(path_to_sentences, "r") as reader: 
        data = json.load(reader) 
        for line in data: 
            l = line.rstrip('\r\n')
            sentences.append(l) 
    return sentences


def load_all_cluster_data(clusters_path): 
    """
    Given a path to where the cluster data is stored, the function 
    returns a dictionary where each key is a cluster id, and each value 
    is a list of tuples consisting of tokens, sentence ids, and token ids 
    (corresponding to the cluster)

    Parameters
    ----------
    clusters_path: str :
        A path to where the cluster data is stored
        
    Returns
    -------
    clusters: Dict
        A dictionary containing the cluster data. 
    """
    clusters = defaultdict(list)
    with open(clusters_path) as fp:
        for line_idx, line in enumerate(fp):
            token, _, sentence_idx, token_idx, cluster_idx = line.strip().rsplit("|||")

            sentence_idx = int(sentence_idx)
            token_idx = int(token_idx)
            cluster_idx = int(cluster_idx)

            clusters["c" + str(cluster_idx)].append((token, sentence_idx, token_idx))
    return clusters





