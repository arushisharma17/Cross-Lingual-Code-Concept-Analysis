import argparse
import os
import json
from collections import defaultdict
import google.generativeai as genai
import time
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import dotenv
dotenv.load_dotenv()

def read_java_in_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return [line.strip() for line in file]

def process_cluster_file(file_path, java_in_lines):
    clusters_data = defaultdict(lambda: {"tokens": set(), "Context Sentences": set()})
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            stripped_line = line.strip()
            if not stripped_line:  # Skip empty lines
                continue
                
            try:
                # Split by ||| to get parts
                parts = stripped_line.split('|||')
                if len(parts) != 5:  # We expect exactly 5 parts based on the format
                    print(f"Unexpected number of parts ({len(parts)}): {stripped_line}")
                    continue
                    
                token = parts[0].strip()
                sentence_id = int(parts[2])
                cluster_id = parts[-1].strip()
                
                # Add data to cluster
                clusters_data[cluster_id]["tokens"].add(token)
                if 0 <= sentence_id < len(java_in_lines):
                    clusters_data[cluster_id]["Context Sentences"].add(java_in_lines[sentence_id])
                    
            except (IndexError, ValueError) as e:
                print(f"Error processing line '{stripped_line}': {str(e)}")
                continue

    # Convert sets to lists for JSON serialization
    result = {}
    for cluster_id in clusters_data:
        result[cluster_id] = {
            "tokens": list(clusters_data[cluster_id]["tokens"]),
            "Context Sentences": list(clusters_data[cluster_id]["Context Sentences"])
        }

    # Debug print
    print(f"Total clusters found: {len(clusters_data)}")
    print(f"First few cluster IDs: {sorted(list(clusters_data.keys()))[:10]}")
    print(f"Last few cluster IDs: {sorted(list(clusters_data.keys()))[-10:]}")

    return result

def label_clusters(clusters_data):
    # Initialize Gemini API
    genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
    
    generation_config = {
        "temperature": 0.3,
        "top_p": 0.4,
        "top_k": 40,
        "max_output_tokens": 300,
        "response_mime_type": "application/json",
    }

    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        generation_config=generation_config,
        system_instruction="You are a C++ Software Developer experienced in analyzing code."
    )

    final_output = []
    max_retries = 10
    retry_delay = 2  # seconds

    # Updated safety settings
    safety_settings = {
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    }

    # Print total clusters to process
    print(f"Processing {len(clusters_data)} clusters found in the file")
    
    # Only process clusters that exist in the data
    for index, (cluster_id, cluster_info) in enumerate(clusters_data.items()):
        print(f"Processing cluster {cluster_id} ({index + 1}/{len(clusters_data)})")
        
        # Rate limiting
        if index > 0 and index % 2000 == 0:
            print("Pausing for 60 seconds due to rate limit...")
            time.sleep(61)

        tokens_list = cluster_info["tokens"]
        context_sentences = cluster_info["Context Sentences"]

        # Prepare the user message
        user_message = f"""
        You are analyzing a cluster of C++ tokens and their context sentences. Each cluster has one or more unique tokens. Your task is to identify the syntactic role or syntactic function these tokens play within the context of the provided sentences. Also focus on the semantic significance of the tokens and what is being achieved in the code.

        Guidelines for Analysis:
        1. **Tokens**: Review the provided tokens. Consider their role in C++ programming (e.g., keywords, operators, identifiers,etc).
        2. **Context Sentences**: Examine the context sentences to understand the usage of the tokens and the code. Look for patterns or common structures.
        3. **Concise Syntactic Label**: Choose a descriptive label that accurately describes the syntactic function or syntactic role of the tokens in the code. Use specific terminology where applicable (e.g., Object,Dot operator,methods) . Avoid generic terms.
        4. **Semantic Tags**: Provide 3-5 semantic tags that accurately describe the key functionality and purpose of the code. These tags should reflect the overall functionality and purpose of the code. Avoid using the same tag twice and avoid generic tags. Aim for detailed tags. Examples of good tags include "Concurrency Control", "Data Serialization", "Error Handling".
        5. **Description**: Provide a concise justification for the syntactic label and semantic tags you have chosen. Explain why these tokens and sentences are significant in the context of Java programming.
        6. **Important**: For special characters or symbols, provide specific syntactic labels (e.g., '(' as 'Opening Parenthesis', ')' as 'Closing Parenthesis'). PLEASE FOLLOW THESE SPECIAL CHARACTER GUIDELINES STRICTLY.
        7. **Important**: The examples below are for Java and are just for reference. Just understand the context and the tokens and provide the C++ labels and tags accordingly.
        
         ## Examples from Previous Java Clusters :
         1. 
         ** Tokens :** ` returnBuffer , concatBuffer `
         ** Context Sentences :**
         - returnBuffer . append ( minParam );
         - returnBuffer . append ( FieldMetaData . Decimal . SQ_CLOSE );
         - StringBuffer concatBuffer = new StringBuffer () ;
         - concatBuffer . append ( toAdd );

          ** Syntactic Label :** Object
          ** Semantic Tags :** Stringbuilder, Form management, Networking
          ** Description :** They're all stringbuilder objects that create strings for form management and networking
         
         2. 
         ** Tokens :** `.`
         ** Context Sentences :**
         - if ( message . getEndpoint ( ) . getEntityType ( ) == this . endpoint . getEntityType ( ) )
         - if ( parameters . isEmpty ( ) && parameter . isRawValue ( ) )
         - . flatMap ( x -> stream ( x . getElements ( ) ) . map ( LinkHeader :: new ) )
         - Template template = templateEngine . getTemplate ( meta . getTemplateFile ( ) ) ;

          ** Syntactic Label :** Dot Operator
          ** Semantic Tags :** Function Calls, Configuration, method invocation
          ** Description :** Dot operator often used to call functions related to configuration .
        
        ## Label this C++ cluster now :
        Tokens: {', '.join(tokens_list)}
        All Context Sentences:
        {chr(10).join([f"{i + 1}. {sentence}" for i, sentence in enumerate(context_sentences)])}
        
        Give your response in this JSON format strictly :
        {{
            "Syntactic Label": "Your concise label here",
            "Semantic Tags": [
                "Tag1",
                "Tag2",
                "Tag3",
                "Tag4",
                "Tag5"
            ],
            "Description": "Your description here."
        }}
        
        Ensure your response is in valid JSON format and includesonly the JSON object and nothing else.
        """

        success = False
        attempts = 0
        
        while not success and attempts < max_retries:
            try:
                # Update the API call with proper safety settings
                response = model.generate_content(user_message, safety_settings=safety_settings)
                response_json = json.loads(response.text)
                
                # Validate required fields exist
                required_fields = ['Syntactic Label', 'Semantic Tags', 'Description']
                if not all(field in response_json for field in required_fields):
                    raise ValueError("Missing required fields in response")
                
                # Validate Semantic_Tags is a list with at least 3 items
                if not isinstance(response_json['Semantic Tags'], list) or len(response_json['Semantic Tags']) < 3:
                    raise ValueError("Semantic_Tags must be a list with at least 3 items")

                final_output.append({
                    "c" + cluster_id: {
                        "Unique tokens": tokens_list,
                        "Syntactic Label": response_json['Syntactic Label'],
                        "Semantic Tags": response_json['Semantic Tags'],
                        "Description": response_json['Description']
                    }
                })
                success = True
                print(f"Successfully processed cluster {cluster_id} on attempt {attempts + 1}")
                
            except (json.JSONDecodeError, ValueError, KeyError) as e:
                attempts += 1
                print(f"Error processing cluster {cluster_id} (Attempt {attempts}/{max_retries}): {str(e)}")
                if attempts < max_retries:
                    print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    print(f"Failed to process cluster {cluster_id} after {max_retries} attempts. Using fallback values.")
                    # Print the response JSON for failed attempts
                    print(f"Response JSON: {response.text}")
                    # Provide fallback values for failed clusters
                    final_output.append({
                        "c" + cluster_id: {
                            "Unique tokens": tokens_list,
                            "Syntactic Label": "Unknown",
                            "Semantic Tags": ["Unknown"],
                            "Description": "Failed to classify due to processing error."
                        }
                    })
            except Exception as e:
                print(f"Unexpected error for cluster {cluster_id}: {str(e)}")
                attempts += 1
                if attempts < max_retries:
                    print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    print(f"Failed to process cluster {cluster_id} after {max_retries} attempts. Using fallback values.")
                    final_output.append({
                        "c" + cluster_id: {
                            "Unique tokens": tokens_list,
                            "Syntactic Label": "Unknown",
                            "Semantic Tags": ["Unknown"],
                            "Description": "Failed to classify due to processing error."
                        }
                    })

    return final_output

def main():
    parser = argparse.ArgumentParser(description="Assign labels to clusters using Gemini API")
    parser.add_argument("--sentence-file", required=True, help="Path to the sentence file")
    parser.add_argument("--model-dir", required=True, help="Path to the model directory containing layer folders")
    parser.add_argument("--dir-extension", required=True, help="Path to the cluster files")
    parser.add_argument("--component", choices=['encoder', 'decoder'], required=True, help="Whether to process encoder or decoder clusters")
    parser.add_argument("--start-layer", type=int, required=True, help="Layer number to start from")
    parser.add_argument("--end-layer", type=int, required=True, help="Layer number to end at")
    args = parser.parse_args()

    java_in_lines = read_java_in_file(args.sentence_file)
    
    # Process only layers 5-9
    layer_dirs = sorted(d for d in os.listdir(args.model_dir) 
                       if os.path.isdir(os.path.join(args.model_dir, d)) 
                       and d.startswith("layer")
                       and args.start_layer <= int(d.replace("layer", "")) <= args.end_layer)

    if not layer_dirs:
        print(f"No layer directories found in {args.model_dir}")
        return

    for layer_dir in layer_dirs:
        print(f"\nProcessing {layer_dir}")
        layer_path = os.path.join(args.model_dir, layer_dir,args.dir_extension)

        # Look for component-specific cluster files directly in layer directory
        cluster_files = [f for f in os.listdir(layer_path) 
                        if f.startswith(f"{args.component}-clusters") and f.endswith(".txt")]
        
        print(f"Found {args.component} cluster files: {cluster_files}")
        
        if cluster_files:
            print(f"Processing file: {cluster_files[0]}")
            process_single_cluster_file(layer_path, cluster_files[0], java_in_lines, args.component)
        else:
            print(f"No matching {args.component} cluster files found in {layer_path}")


def process_single_cluster_file(directory, filename, java_in_lines, component):
    """Helper function to process a single cluster file and save its results"""
    cluster_file = os.path.join(directory, filename)
    print(f"Processing {component} cluster file: {cluster_file}")
    
    # Add debug prints
    print(f"Reading clusters from: {cluster_file}")
    clusters_data = process_cluster_file(cluster_file, java_in_lines)
    print(f"Found {len(clusters_data)} clusters")
    
    final_output = label_clusters(clusters_data)
    print(f"Generated labels for {len(final_output)} clusters")
    
    json_output_path = os.path.join(directory, f'{component}_gemini_labels.json')
    with open(json_output_path, 'w') as f:
        json.dump(final_output, f, indent=2)
    print(f"Saved JSON to {json_output_path}")

if __name__ == "__main__":
    main()