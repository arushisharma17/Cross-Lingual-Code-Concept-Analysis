from flask import Flask, redirect, render_template, request, url_for
from collections import Counter
import argparse 
from pathlib import Path
from utils import read_cluster_data, read_annotations, read_sentences, load_all_cluster_data
app = Flask(__name__) 
import json
import glob


@app.route("/")
def index(): 
   return redirect(url_for("get_cluster", cluster_id=0, layer_id=0)) 


@app.route('/cluster/<layer_id>/<cluster_id>', methods = ['GET'])
def get_cluster(cluster_id, layer_id):
    cluster_id = int(cluster_id)
    temp = "c" + str(cluster_id)
    clusters_path = Path(DATA_PATH) / f"layer{layer_id}" / "clusters-*.txt"
    annotations_path = Path(DATA_PATH) / f"layer{layer_id}" / "annotations.json"
        
    # Check if the parent directory for the annotations file exists, and create it if not
    annotations_path.parent.mkdir(parents=True, exist_ok=True)

    cluster_files = glob.glob(str(clusters_path))
    if not cluster_files:
        return f"<p>No cluster files found for layer {layer_id}</p>", 404
    clusters_path = cluster_files[0]


    split_by_slash = clusters_path.split('/')
    split_by_dash = split_by_slash[-1].split('-')
    cluster_number = int(split_by_dash[1].split('.')[0])
    
    # Load cluster data and annotations
    cluster_to_words = read_cluster_data(clusters_path)
    annotations = read_annotations(annotations_path, cluster_number)

    if temp not in list(cluster_to_words.keys()): 
        return f"<p> Invalid cluster ID {temp} </p>"

    words = cluster_to_words[temp]
    label = annotations[temp]
    
    word_frequencies = list(Counter(words).items())

    return render_template("display.html", word_frequencies=word_frequencies, label=label, cluster_id=cluster_id, layer_id=layer_id, model=MODEL)

def get_cluster_helper(cluster_id, layer_id, file_path):
    cluster_id = int(cluster_id)
    temp = "c" + str(cluster_id)
    clusters_path = Path(DATA_PATH) / f"layer{layer_id}" / file_path
    annotations_path = Path(DATA_PATH) / f"layer{layer_id}" / "annotations.json"
        
    # Check if the parent directory for the annotations file exists, and create it if not
    annotations_path.parent.mkdir(parents=True, exist_ok=True)

    cluster_files = glob.glob(str(clusters_path))
    if not cluster_files:
        return f"<p>No cluster files found for layer {layer_id}</p>", 404
    clusters_path = cluster_files[0]


    cluster_number = int(clusters_path.split('-')[-1].split('.')[0])
    
    # Load cluster data and annotations
    cluster_to_words = read_cluster_data(clusters_path)
    annotations = read_annotations(annotations_path, cluster_number)

    if temp not in list(cluster_to_words.keys()): 
        return f"<p> Invalid cluster ID {temp} </p>"

    words = cluster_to_words[temp]
    label = annotations[temp]
    
    word_frequencies = list(Counter(words).items())

    return (word_frequencies, label)

@app.route("/sentences", methods=['POST']) 
def get_sentences(): 
    cluster_id = request.json["cluster_id"]
    word = request.json["word"] 
    layer_id = request.json["layer_id"]

    word = word.strip() 
    temp = "c" + str(cluster_id)

    clusters_path = Path(DATA_PATH) / f"layer{layer_id}" / "clusters-500.txt"

    clusters = load_all_cluster_data(clusters_path=clusters_path)

    if temp not in list(clusters.keys()): 
        return {"success": True, "sentences": local_sentences}
    
    local_sentences = [[SENTENCES[sentence_idx], token_idx] for token, sentence_idx, token_idx in clusters[f"c{cluster_id}"] if token == word]

    return {"success": True, "sentences": local_sentences}

    
@app.route('/compare/<layer_id>/<encoder_cluster_id>/<decoder_cluster_id>', methods=['GET'])
def get_compare(layer_id, encoder_cluster_id, decoder_cluster_id):

    (encoder_word_frequencies, label) = get_cluster_helper(encoder_cluster_id, layer_id, "encoder-clusters-kmeans-*.txt")
    (decoder_word_frequencies, label) = get_cluster_helper(decoder_cluster_id, layer_id, "decoder-clusters-kmeans-*.txt")
    
    return render_template("compare.html", 
                           encoder_word_frequencies=encoder_word_frequencies, 
                           encoder_cluster_id=encoder_cluster_id, 
                           
                           decoder_word_frequencies=decoder_word_frequencies, 
                           decoder_cluster_id=decoder_cluster_id, 
                           
                           label=label, 
                           layer_id=layer_id, 
                           model=MODEL)


if __name__=="__main__": 

    parser = argparse.ArgumentParser()
    parser.add_argument("-dp", "--data_path", help="path to where the data was downloaded")
    parser.add_argument("-p", "--port", default="9090", help="port used to run the app")
    parser.add_argument("-hs", "--host", default="0.0.0.0", help="host used to run the app")
    args=parser.parse_args()

    DATA_PATH=args.data_path
    PORT=args.port
    HOST=args.host

    sentences_path = Path(DATA_PATH) / "sentences.json"
    # SENTENCES = read_sentences(sentences_path)
    MODEL = DATA_PATH.split("/")[-1]


    app.run(host=HOST, debug=True, port= PORT)


