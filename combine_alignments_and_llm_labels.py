import json
import argparse
from pathlib import Path

def load_json(file_path: Path) -> dict:
    """Load JSON file and return dictionary."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def combine_alignments_with_labels(layer: int) -> None:
    """Combine alignment results with Gemini labels for a specific layer."""
    
    # Hardcoded paths
    base_path = Path("C:/Users/91917/Desktop/Research/Cross-Lingual-Code-Concept-Analysis")
    
    # Alignment file path
    alignment_file = base_path / f"LLM_labelling/t5/java_cs/layer{layer}/cluster_alignments.json"
    
    # Encoder/Decoder labels paths
    encoder_labels_file = base_path / f"LLM_labelling/t5/java_cs/layer{layer}/encoder_gemini_labels.json"
    decoder_labels_file = base_path / f"LLM_labelling/t5/java_cs/layer{layer}/decoder_gemini_labels.json"
    
    # Output directory
    output_dir = base_path / f"LLM_labelling/t5/java_cs/layer{layer}"
    
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
    for src_cluster, target_clusters in alignments.items():
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
            
            # Add decoder cluster information
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
    parser.add_argument("--layer", type=int, required=True,
                      help="Layer number to process")
    
    args = parser.parse_args()
    combine_alignments_with_labels(args.layer)

if __name__ == "__main__":
    main()