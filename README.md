# **LongRAG** 
This repo contains the code for "LongRAG: Enhancing Retrieval-Augmented Generation
with Long-context LLMs". 

### Datasets and Models
Our dataset are all available at Huggingface.

🤗  https://huggingface.co/datasets/TIGER-Lab/LongRAG

## **Table of Contents**
- [Introduction](#introduction)
- [Installation](#installation)
- [Corpus Preparation](#corpus)
- [Long Retriever](#long-retriever)
- [License](#license)
- [Citation](#citation)


## **Introduction**
In order to alleviate the imbalance burden between the retriever and reader of the RAG framework, 
we propose a new framework LongRAG, consisting of a ‘long retriever’ and a ‘long reader’.

## **Installation**

Clone this repository and install the required packages:
```bash
git clone https://github.com/TIGER-AI-Lab/LongRAG.git
cd LongRAG
pip install -r requirements.txt
```

## **Corpus Preparation**

We first preprocess Wikipedia raw data. 
The processed Wikipedia data is "Wiki-NQ" and "Wiki-HotpotQA" subset in our huggingface.

## **License**
Please check out the license of each subset we use in our work.

| Dataset Name 	 | License Type   	               |
|----------------|--------------------------------|
| NQ        	    | Apache License 2.0           	 |
| HotpotQA    	  | CC BY-SA 4.0 License           |


## **Citation**

Please cite our paper if you use our data, model or code. Please also kindly cite the original dataset papers. 

```
@article{longrag,
  title={LongRAG: Enhancing Retrieval-Augmented Generation with Long-context LLMs},
  author={Ziyan Jiang, Xueguang Ma, Wenhu Chen},
  journal={arXiv preprint},
  year={2023}
}
```

