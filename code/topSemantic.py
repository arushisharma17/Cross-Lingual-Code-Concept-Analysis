import json
import os
import argparse
from collections import Counter
from typing import Dict, List

def load_json_files(base_dir: str, extension: str) -> Dict[int, Dict[str, List[str]]]:
    """
    Load all encoder/decoder JSON files from layer directories and extract semantic tags.
    Skip layers where files don't exist.
    """
    all_labels = {}
    
    # Iterate through each layer directory
    for layer in range(13):  # 0-12 layers
        layer_dir = os.path.join(base_dir, f"layer{layer}", extension)
        if not os.path.exists(layer_dir):
            continue
            
        has_valid_files = False
        layer_data = {'encoder': [], 'decoder': []}
        
        # Look for encoder and decoder gemini label files
        for file_type in ['encoder', 'decoder']:
            json_file = f"{file_type}_gemini_labels.json"
            file_path = os.path.join(layer_dir, json_file)
            
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        # Extract semantic tags from each cluster
                        for cluster in data:
                            for cluster_data in cluster.values():
                                if isinstance(cluster_data, dict) and 'Semantic Tags' in cluster_data:
                                    layer_data[file_type].extend(cluster_data['Semantic Tags'])
                        has_valid_files = True
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
                    continue
        
        # Only add layer data if valid files were found and processed
        if has_valid_files:
            all_labels[layer] = layer_data
    
    return all_labels

def analyze_semantic_tags(base_dir: str, extension: str, top_n: int = 20) -> None:
    """
    Analyze semantic tags across all layers and save top N tags for encoder and decoder.
    """
    # Load all labels from JSON files
    layer_labels = load_json_files(base_dir, extension)
    
    if not layer_labels:
        print("No valid label files found in any layer")
        return
        
    # Combine all encoder and decoder labels across all layers
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
    
    # Save results to JSON files in the model directory
    encoder_output = os.path.join(base_dir, 'top_encoder_semantic_tags.json')
    decoder_output = os.path.join(base_dir, 'top_decoder_semantic_tags.json')
    
    with open(encoder_output, 'w', encoding='utf-8') as f:
        json.dump(top_encoder_tags, f, indent=2)
    
    with open(decoder_output, 'w', encoding='utf-8') as f:
        json.dump(top_decoder_tags, f, indent=2)
     
    print(f"Top {top_n} encoder tags saved to {encoder_output}")
    print(f"Top {top_n} decoder tags saved to {decoder_output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze semantic tags across all layers')
    parser.add_argument('--model-dir', type=str, required=True, 
                       help='Base directory containing layer directories')
    parser.add_argument('--dir-extension', type=str, required=True, 
                       help='Extension path within each layer directory')
    parser.add_argument('--top-n', type=int, default=20,
                       help='Number of top tags to save (default: 20)')
    
    args = parser.parse_args()
    
    try:
        analyze_semantic_tags(args.model_dir, args.dir_extension, args.top_n)
    except Exception as e:
        print(f"Error analyzing semantic tags: {e}")
