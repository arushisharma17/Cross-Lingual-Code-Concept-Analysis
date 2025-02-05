# TransformersConceptNet
Code and Data Associated with **Can LLMs facilitate interpretation of pre-trained language models?** 

# Data 
The data can be downloaded from the [project page](https://neurox.qcri.org/projects/transformers-concept-net/)

# Data Description 

We use agglomerative hierarchical clustering over contextualized representations and then annotate these concepts using GPT annotations. Our study involved several 12-layered transformer models, including BERT-cased, RoBERTa, XLNet, and ALBERT. For more details, please refer to the paper

# Load data into display 

Clone this repository: 
```
git clone https://github.com/qcri/TransformersConceptNet.git
cd TransformersConceptNet
```
Create and activate virtual environment: 
```
python -m venv .envs/tcn
source .envs/tcn/bin/activate
```
Install requirements: 
```
pip install -r requirements.txt
```
Start the webapp using the following command:

```bash
python -u app.py -d <path-to-downloaded-data>
```

and visit http://localhost:8080 in your browser. The port and hostname can be passed as additional arguments to `app.py`.

# Concept probing and Neuron Analysis

We used the NeuroX package to train linear probes and perform neuron analysis. You can view the documentation of the package [here](https://neurox.readthedocs.io/en/latest/index.html)

# Todo 

- [ ] Add cross architectural comparison between the models in the display

# Citation

If you used the dataset please cite 

```
@inproceedings{mousi2023llms,
      title = "Can LLMs Facilitate Interpretation of Pre-trained Language 
      Models?",
      author = "Mousi, Basel  and
      Durrani, Nadir  and 
      Dalvi, Fahim", 
      booktitle = "Proceedings of the 2023 Conference on Empirical Methods 
      in Natural Language Processing",
      month = dec, 
      year = "2023", 
      publisher = "Association for Computational Linguistics", 
      url = "https://browse.arxiv.org/pdf/2305.13386.pdf"
}
```
