import json
import argparse
from pathlib import Path
import os
import time

def load_json(file_path: Path) -> dict:
    """Load JSON file and return dictionary."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def combine_alignments_with_labels(layer: int, alignment_file: Path, encoder_labels_file: Path, 
                                decoder_labels_file: Path, output_dir: Path) -> None:
    """Combine alignment results with Gemini labels for a specific layer.
    
    Args:
        layer: Layer number being processed
        alignment_file: Path to cluster alignments file
        encoder_labels_file: Path to encoder Gemini labels
        decoder_labels_file: Path to decoder Gemini labels
        output_dir: Directory to save output
    """
    # Load files
    alignments = load_json(alignment_file)
    encoder_labels = load_json(encoder_labels_file)
    decoder_labels = load_json(decoder_labels_file)
    
    # Create combined structure
    combined_results = {
        "layer": layer,
        "alignments": {}
    }
    
    # Process each source (encoder) cluster
    for src_cluster, alignment_info in alignments.items():
        src_cluster_id = f"c{src_cluster}"  # Format to match Gemini labels format
        
        # Get encoder cluster info
        encoder_info = next((item[src_cluster_id] for item in encoder_labels if src_cluster_id in item), None)
        
        if encoder_info:
            combined_results["alignments"][src_cluster_id] = {
                "encoder_cluster": {
                    "id": src_cluster_id,
                    "unique_tokens": encoder_info.get("Unique tokens", []),
                    "syntactic_label": encoder_info.get("Syntactic Label", ""),
                    "semantic_tags": encoder_info.get("Semantic Tags", []),
                    "description": encoder_info.get("Description", "")
                },
                "aligned_decoder_clusters": []
            }
            
            # Add decoder cluster information using alignment info
            target_clusters = alignment_info.get("aligned_clusters", [])
            for target_cluster in target_clusters:
                target_cluster_id = f"c{target_cluster}"
                decoder_info = next((item[target_cluster_id] for item in decoder_labels if target_cluster_id in item), None)
                
                if decoder_info:
                    decoder_cluster_info = {
                        "id": target_cluster_id,
                        "unique_tokens": decoder_info.get("Unique tokens", []),
                        "syntactic_label": decoder_info.get("Syntactic Label", ""),
                        "semantic_tags": decoder_info.get("Semantic Tags", []),
                        "description": decoder_info.get("Description", "")
                    }
                    combined_results["alignments"][src_cluster_id]["aligned_decoder_clusters"].append(decoder_cluster_info)
    
    # Save combined results
    output_file = output_dir / f"Alignments_with_LLM_labels_layer{layer}.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(combined_results, f, indent=2, ensure_ascii=False)
    
    print(f"Combined results saved to: {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Combine cluster alignments with Gemini labels")
    parser.add_argument("--model-dir", required=True, help="Path to the model directory containing layer folders")
    parser.add_argument("--dir-extension", required=True, help="Path to the cluster files")
    parser.add_argument("--start-layer", type=int, required=True, help="Layer number to start from")
    parser.add_argument("--end-layer", type=int, required=True, help="Layer number to end at")
    
    args = parser.parse_args()
    
    # Process layers in the range
    layer_dirs = sorted(d for d in os.listdir(args.model_dir) 
                       if os.path.isdir(os.path.join(args.model_dir, d)) 
                       and d.startswith("layer")
                       and args.start_layer <= int(d.replace("layer", "")) <= args.end_layer)

    if not layer_dirs:
        print(f"No layer directories found in {args.model_dir}")
        return

    for layer_dir in layer_dirs:
        print(f"\nProcessing {layer_dir}")
        layer_path = os.path.join(args.model_dir, layer_dir, args.dir_extension)
        
        # All files should be in the same directory
        layer_num = int(layer_dir.replace("layer", ""))
        alignment_file = Path(layer_path) / "cluster_alignments.json"
        encoder_labels = Path(layer_path) / "encoder_gemini_labels.json"
        decoder_labels = Path(layer_path) / "decoder_gemini_labels.json"
        
        if not all(f.exists() for f in [alignment_file, encoder_labels, decoder_labels]):
            print(f"Missing required files in {layer_path}. Skipping layer {layer_num}.")
            continue
        
        # Process the layer
        combine_alignments_with_labels(
            layer_num,
            alignment_file,
            encoder_labels,
            decoder_labels,
            Path(layer_path)
        )
        

if __name__ == "__main__":
    main()