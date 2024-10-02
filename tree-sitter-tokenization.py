import numpy as np
import re
import collections
from typing import Pattern, Callable, Dict, List
from tree_sitter import Language, Parser
import tree_sitter_java as tsjava
import graphviz

# Set up Java language parser
JAVA_LANGUAGE = Language(tsjava.language())
parser = Parser(JAVA_LANGUAGE)

# Function to read source code from a file
def read_source_code(file_path: str) -> bytes:
    with open(file_path, 'r') as file:
        source_code = file.read()
    return source_code.encode('utf-8')

# Example: Generating a sample Java file
sample_code = '''
public class HelloWorld {
    public static void main(String[] args) {
        System.out.println("Hello, World!");
    }
}
'''

# Write the sample Java code to a file
with open("sample_input.java", "w") as file:
    file.write(sample_code)

# Read the source code from the file
source_code = read_source_code("sample_input.java")

# Parse the source code
tree = parser.parse(source_code)
root_node = tree.root_node

# Extract tokens from leaf nodes only
def extract_leaf_tokens(node):
    tokens = []
    if len(node.children) == 0:  # If the node is a leaf node
        tokens.append((node.type, node.text.decode('utf-8')))
    else:
        for child in node.children:
            tokens.extend(extract_leaf_tokens(child))
    return tokens

# Extract tokens and labels from the AST
tokens = extract_leaf_tokens(root_node)

# Write tokens to a file
with open("tokens.txt", "w") as token_file:
    for _, token_text in tokens:
        token_file.write(f"{token_text}\n")

# Write labels to a file
with open("labels.txt", "w") as label_file:
    for token_type, _ in tokens:
        label_file.write(f"{token_type}\n")

# Print tokens and labels to inspect their types and contents
print("Extracted Tokens and Labels:")
for token_type, token_text in tokens:
    print(f"Type: {token_type}, Text: {token_text}")

# Optional: Visualize the AST using Graphviz
def visualize_ast(node, graph, parent_id=None):
    node_id = str(id(node))
    label = f"{node.type} [{node.start_point}-{node.end_point}]"
    graph.node(node_id, label)
    if parent_id:
        graph.edge(parent_id, node_id)
    for child in node.children:
        visualize_ast(child, graph, node_id)

graph = graphviz.Digraph(format="png")
visualize_ast(root_node, graph)
graph.render("java_ast")
