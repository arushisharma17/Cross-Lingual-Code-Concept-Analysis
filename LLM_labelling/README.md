# Gemini Labelling

TO run the labelling script, you need to have the following:

    1. Model directory with all layers having their encoder and decoder clusters
    2. Input file with encoder sentences - input.in
    3. Output file with decoder sentences - label.out

Directory structure:

```
labelling/
    t5/
        java_cs/
                layer1/
                    encoder-clusters.txt
                    decoder-clusters.txt
                layer2/
                    encoder-clusters.txt
                    decoder-clusters.txt
        input.in
        label.out
    
    coderosetta/
        java_cs/
                layer1/
                    encoder-clusters.txt
                    decoder-clusters.txt
                layer2/
                    encoder-clusters.txt
                    decoder-clusters.txt
        input.in
        label.out
```

Command to run the labelling script:

```bash
python LLM_labelling/gemini_labelling_<language>.py --sentence-file <sentence_file> --model-dir <model_dir> --component <component> --start-layer <start_layer> --end-layer <end_layer>
```
Example for Java encoder:
```bash
python LLM_labelling/gemini_labelling_java.py --sentence-file LLM_labelling/t5/java_cs/input.in  --model-dir LLM_labelling/t5/java_cs --component encoder --start-layer 0 --end-layer 12
```
Example for C# decoder:
```bash
python LLM_labelling/gemini_labelling_csharp.py --sentence-file LLM_labelling/t5/java_cs/label.out  --model-dir LLM_labelling/t5/java_cs --component decoder --start-layer 0 --end-layer 12
```
