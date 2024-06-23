# **LongRAG** 
This repo contains the code for "LongRAG: Enhancing Retrieval-Augmented Generation
with Long-context LLMs". We are still in the process to polish our repo.

### Datasets and Models
Our dataset are all available at Huggingface.

🤗  https://huggingface.co/datasets/TIGER-Lab/LongRAG

## **Table of Contents**
- [Introduction](#introduction)
- [Quick Start](#qucick-start)
- [Installation](#installation)
- [Corpus Preparation](#corpus)
- [Long Retriever](#long-retriever)
- [Long Reader](#long-reader)
- [Evaluation](#eval)
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

***Wikipedia preprocess:***
We first preprocess Wikipedia raw data. 
The processed Wikipedia data is "Wiki-NQ" and "Wiki-HotpotQA" subset in our huggingface.
"Wiki-NQ" is the processed Wikipedia dumps from December 20, 2018. "Wiki-HotpotQA" is 
the abstract paragraphs from the October 1, 2017, dump.

***Retrieval Corpus:*** By grouping multiple related documents, we can construct long 
retrieval units with more than 4K tokens. This design could also significantly reduce 
the corpus size (number of retrieval units in the corpus). Then, the retriever’s task 
becomes much easier. Additionally, the long retrieval unit will also improve the 
information completeness to avoid ambiguity or confusion.

```bash
sh scripts/group_documents.sh
```

We have released our retrieval corpus.



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
  year={2024}
}
```
