import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import numpy as np
from PIL import Image, ImageDraw
from collections import Counter

def load_json(file_path: Path) -> dict:
    """Load JSON file and return dictionary."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_cluster_text(cluster_info: dict) -> dict:
    """Get frequencies of unique tokens from a cluster."""
    frequencies = {}
    
    # Add unique tokens with high weight
    if "unique_tokens" in cluster_info:
        for token in cluster_info["unique_tokens"]:
            frequencies[token] = 200  # High weight for prominence
        print("Unique tokens found:", cluster_info["unique_tokens"])
    else:
        print("No unique tokens found.")
    
    print("Final frequencies:", frequencies)
    return frequencies

def create_wordcloud_for_pair(encoder_cluster: dict, decoder_cluster: dict, 
                            output_path: Path, encoder_id: str, decoder_id: str) -> None:
    """Create word cloud visualization for a single encoder-decoder cluster pair."""
    
    print(f"\nProcessing pair: Encoder {encoder_id} -> Decoder {decoder_id}")
    
    # Get frequency dictionaries for each cluster
    encoder_freq = get_cluster_text(encoder_cluster)
    decoder_freq = get_cluster_text(decoder_cluster)
    
    if not encoder_freq or not decoder_freq:
        print(f"Warning: Empty frequencies for encoder {encoder_id} or decoder {decoder_id}")
        return
    
    # Create figure
    plt.figure(figsize=(24, 12))
    
    # Generate word clouds using frequencies
    wc_encoder = WordCloud(
        width=1000, height=800,
        background_color='white',
        contour_width=1,
        contour_color='black',
        min_font_size=12,
        max_font_size=150,
        scale=2,
        relative_scaling=0.5,
        max_words=200,
        colormap='viridis',
        prefer_horizontal=0.9  # Increase horizontal preference for compactness
    ).generate_from_frequencies(encoder_freq)
    
    wc_decoder = WordCloud(
        width=1000, height=800,
        background_color='white',
        contour_width=1,
        contour_color='black',
        min_font_size=12,
        max_font_size=150,
        scale=2,
        relative_scaling=0.5,
        max_words=200,
        colormap='viridis',
        prefer_horizontal=0.9  # Increase horizontal preference for compactness
    ).generate_from_frequencies(decoder_freq)
    
    # Plot encoder word cloud
    plt.subplot(1, 2, 1)
    plt.imshow(wc_encoder, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'Encoder Cluster {encoder_id}', pad=20, fontsize=20)  # Increased font size
    
    # Plot decoder word cloud
    plt.subplot(1, 2, 2)
    plt.imshow(wc_decoder, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'Decoder Cluster {decoder_id}', pad=20, fontsize=20)  # Increased font size
    
    # Save plot
    output_file = output_path / f"wordcloud_encoder{encoder_id}_to_decoder{decoder_id}.png"
    plt.savefig(output_file, bbox_inches='tight', dpi=300, pad_inches=0.5)
    plt.close()
    
    print(f"Created wordcloud for encoder {encoder_id} -> decoder {decoder_id}")

def create_split_wordclouds(layer: int) -> None:
    """Create word cloud visualizations for all encoder-decoder cluster pairs."""
    
    # Hardcoded paths
    base_path = Path("C:/Users/91917/Desktop/Research/Cross-Lingual-Code-Concept-Analysis")
    results_file = base_path / f"LLM_labelling/coderosetta/cpp_cuda/layer{layer}/Alignments_with_LLM_labels_layer{layer}.json"
    output_dir = base_path / f"LLM_labelling/coderosetta/cpp_cuda/layer{layer}/visualizations"
    
    # Skip if results file doesn't exist
    if not results_file.exists():
        print(f"\nSkipping layer {layer} - results file not found: {results_file}")
        return
        
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nLoading results from: {results_file}")
    results = load_json(results_file)
    
    total_pairs = 0
    for encoder_id, cluster_data in results["alignments"].items():
        encoder_cluster = cluster_data.get("encoder_cluster", {})
        
        # Debug print
        print(f"\nProcessing encoder cluster {encoder_id}")
        print(f"Encoder cluster keys: {encoder_cluster.keys()}")
        
        for decoder_cluster in cluster_data.get("aligned_decoder_clusters", []):
            # Debug print
            print(f"Decoder cluster keys: {decoder_cluster.keys()}")
            
            create_wordcloud_for_pair(
                encoder_cluster,
                decoder_cluster,
                output_dir,
                encoder_id,
                decoder_cluster.get("id", "unknown")
            )
            total_pairs += 1
    
    print(f"\nCreated {total_pairs} wordcloud visualizations for layer {layer}")

if __name__ == "__main__":
    # Remove argparse
    print("Generating wordclouds for all layers (0-12)...")
    for layer in range(7):  # 0 to 12 inclusive
        print(f"\nProcessing layer {layer}")
        create_split_wordclouds(layer)
    
    for layer in range(10,13):  # 10 to 12 inclusive
        print(f"\nProcessing layer {layer}")
        create_split_wordclouds(layer)
    
    print("\nCompleted processing all layers") 