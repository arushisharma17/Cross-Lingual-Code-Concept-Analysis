import json
import os
import argparse
from collections import Counter
from typing import Dict, List

def load_json_files(base_dir: str) -> Dict[int, Dict[str, List[str]]]:
    """
    Load all encoder/decoder JSON files from layer directories and extract semantic tags.
    """
    all_labels = {}
    base_dir = os.path.normpath(base_dir)
    
    # Iterate through each layer directory
    for layer in range(13):  # 0-12 layers
        layer_dir = os.path.join(base_dir, f"layer{layer}")
        if not os.path.exists(layer_dir):
            print(f"Warning: Directory not found: {layer_dir}")
            continue
            
        all_labels[layer] = {'encoder': [], 'decoder': []}
        
        # Look for encoder and decoder JSON files
        for file_type in ['encoder', 'decoder']:
            try:
                json_files = [f for f in os.listdir(layer_dir) 
                            if f.endswith('.json') and file_type in f]
                
                for json_file in json_files:
                    file_path = os.path.join(layer_dir, json_file)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            # Extract semantic tags from each cluster
                            for cluster in data:
                                for cluster_data in cluster.values():
                                    if isinstance(cluster_data, dict) and 'Semantic Tags' in cluster_data:
                                        all_labels[layer][file_type].extend(cluster_data['Semantic Tags'])
                    except Exception as e:
                        print(f"Error reading {file_path}: {e}")
            except Exception as e:
                print(f"Error accessing {layer_dir}: {e}")
    
    return all_labels

def analyze_semantic_tags(base_dir: str, top_n: int = 20) -> None:
    """
    Analyze semantic tags and save top N tags for encoder and decoder to JSON files.
    """
    # Load all labels from JSON files
    layer_labels = load_json_files(base_dir)
    
    # Combine all encoder and decoder labels separately
    all_encoder_tags = []
    all_decoder_tags = []
    
    for layer_data in layer_labels.values():
        all_encoder_tags.extend(layer_data['encoder'])
        all_decoder_tags.extend(layer_data['decoder'])
    
    # Count occurrences for each type
    encoder_counter = Counter(all_encoder_tags)
    decoder_counter = Counter(all_decoder_tags)
    
    # Get the top N tags for each type
    top_encoder_tags = dict(encoder_counter.most_common(top_n))
    top_decoder_tags = dict(decoder_counter.most_common(top_n))
    
    # Save results to JSON files
    with open('top_encoder_tags.json', 'w', encoding='utf-8') as f:
        json.dump(top_encoder_tags, f, indent=2)
    
    with open('top_decoder_tags.json', 'w', encoding='utf-8') as f:
        json.dump(top_decoder_tags, f, indent=2)
    
    # Print results for verification
    print("\nTop tags have been saved to:")
    print("- top_encoder_tags.json")
    print("- top_decoder_tags.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze semantic tags across layers')
    parser.add_argument('--dir', type=str, required=True, 
                       help='Base directory containing layer directories')
    
    args = parser.parse_args()
    
    try:
        analyze_semantic_tags(args.dir)
    except Exception as e:
        print(f"Error analyzing semantic tags: {e}")
