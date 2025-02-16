from transformers import AutoTokenizer, EncoderDecoderModel

# Load the CodeRosetta model and tokenizer
model = EncoderDecoderModel.from_pretrained('CodeRosetta/CodeRosetta_cpp2cuda_ft')
tokenizer = AutoTokenizer.from_pretrained('CodeRosetta/CodeRosetta_cpp2cuda_ft')

# Encode the input C++ Code
input_cpp_code = "void add_100 ( int numElements , int * data ) { for ( int idx = 0 ; idx < numElements ; idx ++ ) { data [ idx ] += 100 ; } }"
input_ids = tokenizer.encode(input_cpp_code, return_tensors="pt")

# Set the start token to <CUDA>
start_token = "<CUDA>"
decoder_start_token_id = tokenizer.convert_tokens_to_ids(start_token)

# Generate the CUDA code
output = model.generate(
    input_ids=input_ids, 
    decoder_start_token_id=decoder_start_token_id,
    max_length=256
)

# Decode and print the generated output
generated_code = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_code)
