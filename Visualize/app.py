import streamlit as st
import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import defaultdict
import argparse
import pandas as pd

def load_sentences(model_dir: str, component: str):
    """Load sentences from input.in or label.out based on component"""
    file_name = "input.in" if component == "encoder" else "label.out"
    file_path = os.path.join(model_dir, file_name)
    
    if not os.path.exists(file_path):
        return None
        
    with open(file_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f]

def create_wordcloud(tokens):
    """Create and return a word cloud from tokens"""
    if not tokens:
        return None
        
    # Create frequency dict
    freq_dict = {token: 1 for token in tokens}
    
    wc = WordCloud(
        width=800, height=400,
        background_color='white',
        max_words=100
    ).generate_from_frequencies(freq_dict)
    
    return wc

def display_cluster_info(cluster_data, model_pair: str, layer_number: int, cluster_id: str, sentences=None):
    """Display cluster information including word cloud, metadata and sentences"""
    # Store model_pair in session state to persist across reruns
    if 'model_pair' not in st.session_state:
        st.session_state.model_pair = model_pair
    
    # Word cloud and metadata in columns
    col1, col2 = st.columns([2, 1])
    
    with col1:
        tokens = cluster_data.get("Unique tokens", [])
        wc = create_wordcloud(tokens)
        if wc:
            fig = plt.figure(figsize=(10, 5))
            plt.imshow(wc, interpolation='bilinear')
            plt.axis('off')
            st.pyplot(fig)
            plt.close(fig)
            
    with col2:
        st.write("### Metadata")
        st.write(f"**Syntactic Label:** {cluster_data.get('Syntactic Label', 'N/A')}")
        st.write("**Semantic Tags:**")
        for tag in cluster_data.get('Semantic Tags', []):
            st.write(f"- {tag}")
        st.write(f"**Description:** {cluster_data.get('Description', 'N/A')}")

    # Display context sentences
    if sentences:
        st.write("---")
        st.write("### Context Sentences")
        
        with st.container():
            for sent_info in sentences:
                tokens = sent_info["sentence"].split()
                html = create_sentence_html(tokens, sent_info)
                st.markdown(html, unsafe_allow_html=True)

def load_cluster_sentences(model_dir: str, layer: int, component: str, encoder_file: str, decoder_file: str):
    """Load sentences and their indices from cluster file"""
    # Determine file paths
    cluster_file = os.path.join(
        model_dir,
        f"layer{layer}",
        "extraction_without_filtering",
        "clustering",
        f"{component}-clusters-kmeans-500.txt"
    )
    sentence_file = encoder_file if component == "encoder" else decoder_file
    
    if not os.path.exists(cluster_file):
        st.error(f"Cluster file not found: {cluster_file}")
        return defaultdict(list)
        
    if not os.path.exists(sentence_file):
        st.error(f"Sentence file not found: {sentence_file}")
        return defaultdict(list)

    # Load all sentences first
    with open(sentence_file, 'r', encoding='utf-8') as f:
        all_sentences = [line.strip() for line in f]
    
    # Process cluster file to get sentence mappings
    cluster_sentences = defaultdict(list)
    
    with open(cluster_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('|||')
            if len(parts) == 5:  # Expected format: token|||other|||sent_id|||token_idx|||cluster_id
                token = parts[0].strip()
                sentence_id = int(parts[2])
                token_idx = int(parts[3])
                cluster_id = parts[4].strip()
                
                if 0 <= sentence_id < len(all_sentences):
                    cluster_sentences[f"c{cluster_id}"].append({
                        "sentence": all_sentences[sentence_id],
                        "token": token,
                        "token_idx": token_idx
                    })
    
    return cluster_sentences

def display_aligned_clusters(model_base: str, dir_extension: str, selected_layer: int, encoder_file: str, decoder_file: str):
    """Display aligned encoder-decoder cluster pairs with wordclouds"""
    st.header(f"Aligned Clusters - Layer {selected_layer}")
    
    # Debug: Print paths being accessed
    alignments_file = os.path.join(
        model_base,
        f"layer{selected_layer}",
        dir_extension,
        f"Alignments_with_LLM_labels_layer{selected_layer}.json"
    )
    
    metrics_file = os.path.join(
        model_base,
        f"layer{selected_layer}",
        dir_extension,
        "cluster_alignments.json"
    )
    
    if not os.path.exists(alignments_file):
        st.error(f"No alignment data found at: {alignments_file}")
        return
        
    with open(alignments_file, 'r') as f:
        alignments = json.load(f)
        
    with open(metrics_file, 'r') as f:
        alignment_metrics = json.load(f)

    # Create dropdown options for cluster pairs
    cluster_pairs = []
    for src_cluster_id, cluster_data in alignments.get("alignments", {}).items():
        if isinstance(cluster_data, dict):  # Ensure we have a valid cluster data dictionary
            encoder_cluster = cluster_data.get("encoder_cluster", {})
            encoder_id = encoder_cluster.get("id")
            if encoder_id:
                for decoder_cluster in cluster_data.get("aligned_decoder_clusters", []):
                    decoder_id = decoder_cluster.get("id")
                    if decoder_id:
                        cluster_pairs.append((encoder_id, decoder_id))
    
    if not cluster_pairs:
        st.error("No valid cluster pairs found")
        return

    # Dropdown for cluster selection
    selected_pair_idx = st.selectbox(
        "Select cluster pair",
        range(len(cluster_pairs)),
        format_func=lambda x: f"Encoder {cluster_pairs[x][0]} → Decoder {cluster_pairs[x][1]}"
    )
    
    # Get the selected encoder and decoder IDs
    selected_encoder_id, selected_decoder_id = cluster_pairs[selected_pair_idx]
    
    # The issue might be that the encoder ID in metrics file doesn't include 'c' prefix
    metrics_encoder_id = selected_encoder_id.lstrip('c') if selected_encoder_id.startswith('c') else selected_encoder_id
    
    # Find the corresponding data
    for src_cluster_id, cluster_data in alignments["alignments"].items():
        if cluster_data["encoder_cluster"]["id"] == selected_encoder_id:
            encoder_cluster = cluster_data["encoder_cluster"]
            decoder_cluster = next(
                dc for dc in cluster_data["aligned_decoder_clusters"] 
                if dc["id"] == selected_decoder_id
            )
            break
    
    # Load sentences for both encoder and decoder
    encoder_sentences = load_cluster_sentences(
        model_base,
        selected_layer,
        "encoder",
        encoder_file,
        decoder_file
    )
    
    decoder_sentences = load_cluster_sentences(
        model_base,
        selected_layer,
        "decoder",
        encoder_file,
        decoder_file
    )
    
    # Display clusters side by side with wordclouds
    st.write("### Cluster Details")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("#### Encoder Cluster")
        tokens = encoder_cluster.get('unique_tokens', [])
        wc = create_wordcloud(tokens)
        if wc:
            fig = plt.figure(figsize=(10, 5))
            plt.imshow(wc, interpolation='bilinear')
            plt.axis('off')
            st.pyplot(fig)
            plt.close(fig)
            
        st.write(f"**Syntactic Label:** {encoder_cluster.get('syntactic_label', 'N/A')}")
        st.write("**Semantic Tags:**")
        for tag in encoder_cluster.get('semantic_tags', []):
            st.write(f"- {tag}")
        st.write(f"**Description:** {encoder_cluster.get('description', 'N/A')}")
    
    with col2:
        st.write("#### Decoder Cluster")
        tokens = decoder_cluster.get('unique_tokens', [])
        wc = create_wordcloud(tokens)
        if wc:
            fig = plt.figure(figsize=(10, 5))
            plt.imshow(wc, interpolation='bilinear')
            plt.axis('off')
            st.pyplot(fig)
            plt.close(fig)
            
        st.write(f"**Syntactic Label:** {decoder_cluster.get('syntactic_label', 'N/A')}")
        st.write("**Semantic Tags:**")
        for tag in decoder_cluster.get('semantic_tags', []):
            st.write(f"- {tag}")
        st.write(f"**Description:** {decoder_cluster.get('description', 'N/A')}")
    
    # Display alignment metrics
    st.write("### Alignment Metrics")
    if metrics_encoder_id in alignment_metrics:
        metrics = alignment_metrics[metrics_encoder_id]["metrics"]
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Match Percentage", f"{metrics['match_percentage']:.2%}")
            st.metric("Source Cluster Size", metrics["source_cluster_size"])
        
        with col2:
            st.metric("Aligned Word Count", metrics["aligned_word_count"])
            st.metric("Total Words", metrics["total_words"])
            
        with col3:
            st.metric("Size Threshold", metrics["size_threshold"])
            st.metric("Translation Threshold", metrics["translation_threshold"])
    
    # Display context sentences
    if encoder_sentences.get(encoder_cluster['id']):
        st.write("### Encoder Context Sentences")
        for sent_info in encoder_sentences[encoder_cluster['id']]:
            tokens = sent_info["sentence"].split()
            html = create_sentence_html(tokens, sent_info)
            st.markdown(html, unsafe_allow_html=True)
            
    if decoder_sentences.get(decoder_cluster['id']):
        st.write("### Decoder Context Sentences")
        for sent_info in decoder_sentences[decoder_cluster['id']]:
            tokens = sent_info["sentence"].split()
            html = create_sentence_html(tokens, sent_info)
            st.markdown(html, unsafe_allow_html=True)

def create_sentence_html(tokens, sent_info):
    """Helper function to create HTML for sentence display"""
    html = """
    <div style='font-family: monospace; padding: 10px; margin: 5px 0; background-color: #f5f5f5; border-radius: 5px;'>
        <div style='margin-bottom: 5px;'>"""
    
    for idx, token in enumerate(tokens):
        if idx == sent_info["token_idx"]:
            html += f"<span style='color: #2196F3; font-weight: bold;'>{token}</span> "
        else:
            html += f"{token} "
    
    html += f"""
        </div>
        <div style='color: #666; font-size: 0.9em;'>Token: <code>{sent_info['token']}</code></div>
    </div>
    """
    return html

def display_top_semantic_tags(model_base: str, dir_extension: str):
    """Display most common semantic tags for both encoder and decoder"""
    st.header("Top Semantic Tags Across All Layers")
    
    # Load pre-computed top tags
    encoder_file = os.path.join(model_base, "top_encoder_semantic_tags.json")
    decoder_file = os.path.join(model_base, "top_decoder_semantic_tags.json")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Encoder Top Tags")
        if os.path.exists(encoder_file):
            with open(encoder_file, 'r') as f:
                encoder_tags = json.load(f)
                if encoder_tags:
                    # Convert to format suitable for bar chart
                    if isinstance(encoder_tags, dict):
                        data = pd.DataFrame({
                            'tag': list(encoder_tags.keys()),
                            'count': list(encoder_tags.values())
                        })
                    else:
                        data = pd.DataFrame(encoder_tags, columns=['tag', 'count'])
                    
                    # Sort and get top 20
                    data = data.sort_values('count', ascending=True).tail(20)
                    
                    # Create horizontal bar chart using plotly
                    fig = plt.figure(figsize=(10, 8))
                    plt.barh(data['tag'], data['count'])
                    plt.xlabel('Count')
                    plt.ylabel('Semantic Tags')
                    plt.title('Top 20 Encoder Tags')
                    st.pyplot(fig)
                    plt.close()
        else:
            st.error("No encoder tags file found")
            
    with col2:
        st.subheader("Decoder Top Tags")
        if os.path.exists(decoder_file):
            with open(decoder_file, 'r') as f:
                decoder_tags = json.load(f)
                if decoder_tags:
                    # Convert to format suitable for bar chart
                    if isinstance(decoder_tags, dict):
                        data = pd.DataFrame({
                            'tag': list(decoder_tags.keys()),
                            'count': list(decoder_tags.values())
                        })
                    else:
                        data = pd.DataFrame(decoder_tags, columns=['tag', 'count'])
                    
                    # Sort and get top 20
                    data = data.sort_values('count', ascending=True).tail(20)
                    
                    # Create horizontal bar chart using plotly
                    fig = plt.figure(figsize=(10, 8))
                    plt.barh(data['tag'], data['count'])
                    plt.xlabel('Count')
                    plt.ylabel('Semantic Tags')
                    plt.title('Top 20 Decoder Tags')
                    st.pyplot(fig)
                    plt.close()
        else:
            st.error("No decoder tags file found")

def display_individual_clusters(model_base: str, dir_extension: str, layer: int, component: str, sentences_file: str):
    """Display individual cluster information"""
    st.header(f"{component.title()} Clusters - Layer {layer}")
    
    # Debug: Print paths
    clusters_file = os.path.join(
        model_base,
        f"layer{layer}",
        "extraction_without_filtering",
        "clustering",
        f"{component}-clusters-kmeans-500.txt"
    )
    
    # Load gemini labels
    labels_file = os.path.join(
        model_base,
        f"layer{layer}",
        "extraction_without_filtering",
        "clustering",
        f"{component}_gemini_labels.json"
    )
    
    if not os.path.exists(clusters_file):
        st.error(f"No cluster file found at: {clusters_file}")
        return
        
    # Load cluster information
    cluster_sentences = load_cluster_sentences(
        model_base,
        layer,
        component,
        sentences_file,
        sentences_file
    )
    
    # Load labels if available
    cluster_labels = {}
    if os.path.exists(labels_file):
        with open(labels_file, 'r') as f:
            labels_data = json.load(f)
            for item in labels_data:
                cluster_labels.update(item)
    
    # Create cluster selection
    cluster_ids = sorted(cluster_sentences.keys())
    selected_cluster = st.selectbox(
        "Select Cluster",
        cluster_ids,
        format_func=lambda x: f"Cluster {x}"
    )
    
    if selected_cluster:
        st.write(f"### Cluster {selected_cluster}")
        
        # Create two columns for wordcloud and metadata
        col1, col2 = st.columns([1, 1])
        
        with col1:
            sentences = cluster_sentences[selected_cluster]
            tokens = [sent_info["token"] for sent_info in sentences]
            wc = create_wordcloud(tokens)
            if wc:
                fig = plt.figure(figsize=(6, 3))  # Smaller size
                plt.imshow(wc, interpolation='bilinear')
                plt.axis('off')
                st.pyplot(fig)
                plt.close(fig)
        
        with col2:
            st.write("### Metadata")
            if selected_cluster in cluster_labels:
                labels = cluster_labels[selected_cluster]
                st.write(f"**Syntactic Label:** {labels.get('Syntactic Label', 'N/A')}")
                st.write("**Semantic Tags:**")
                for tag in labels.get('Semantic Tags', []):
                    st.write(f"- {tag}")
                st.write(f"**Description:** {labels.get('Description', 'N/A')}")
            else:
                st.write("No labels available for this cluster")
        
        # Display sentences below
        st.write("### Context Sentences")
        for sent_info in sentences:
            tokens = sent_info["sentence"].split()
            html = create_sentence_html(tokens, sent_info)
            st.markdown(html, unsafe_allow_html=True)

def main():
    parser = argparse.ArgumentParser(description='Code Concept Cluster Explorer')
    parser.add_argument('--model-dir', type=str, required=True, 
                       help='Base directory containing model layers')
    parser.add_argument('--dir-extension', type=str, required=True, 
                       help='Directory extension for each layer')
    parser.add_argument('--encoder-file', type=str, required=True, 
                       help='Path to encoder sentences file')
    parser.add_argument('--decoder-file', type=str, required=True, 
                       help='Path to decoder sentences file')
    args = parser.parse_args()

    st.set_page_config(layout="wide", page_title="Code Concept Explorer")
    st.title("Code Concept Cluster Explorer")
    
    
    # Get available layers
    layers = []
    for d in os.listdir(args.model_dir):
        if d.startswith('layer') and os.path.isdir(os.path.join(args.model_dir, d)):
            try:
                layer_num = int(d.replace('layer', ''))
                layers.append(layer_num)
            except ValueError:
                continue
    
    layers.sort()
    
    if not layers:
        st.error("No layers found in the specified directory")
        return
        
    # Sidebar controls
    st.sidebar.header("Settings")
    selected_layer = st.sidebar.selectbox("Select Layer", layers)
    
    # View selection
    view = st.sidebar.radio("View", ["Individual Clusters", "Aligned Clusters", "Top Semantic Tags Across All Layers"])
    
    if view == "Individual Clusters":
        component = st.sidebar.radio("Component", ["Encoder", "Decoder"])
        display_individual_clusters(
            args.model_dir,
            args.dir_extension,
            selected_layer,
            component.lower(),
            args.encoder_file if component == "Encoder" else args.decoder_file
        )
    elif view == "Top Semantic Tags Across All Layers":
        display_top_semantic_tags(args.model_dir, args.dir_extension)
    else:
        display_aligned_clusters(
            args.model_dir,
            args.dir_extension,
            selected_layer,
            args.encoder_file,
            args.decoder_file
        )

if __name__ == "__main__":
    main()