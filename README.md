# **LongRAG** 
This repo contains the code for "LongRAG: Enhancing Retrieval-Augmented Generation
with Long-context LLMs". <span style="color: red;">We are still in the process to polish our repo.</span>

### Datasets and Models
Our dataset are all available at Huggingface.

🤗  https://huggingface.co/datasets/TIGER-Lab/LongRAG

## **Table of Contents**
- [Introduction](#introduction)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Corpus Preparation](#corpus)
- [Long Retriever](#long-retriever)
- [Long Reader](#long-reader)
- [Evaluation](#evaluation)
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

## **Long Retriever**
We leverage open-sourced dense retrieval toolkit, Tevatron. For all our retrieval experiments. 
The base embedding model we used is bge-large-en-v1.5.
```bash
sh scripts/run_retrieve_tevatron.sh
```

## **Long Reader**
We select Gemini-1.5-Pro and GPT-4o as our long reader given their strong ability
to handle long context input.
The input of the reader is a concatenation of all the long retrieval units from the long retriever.
We have provided the input file in our Huggingface repo. For each dataset(NQ or HotpotQA), there are 
three splits: ``full``, ``subset_1000``, ``subset_100``. We suggest starting with ``subset_100`` for a 
quick start or debugging and using ``subset_1000`` to obtain relatively stable results.

```bash
sh scripts/run_eval_qa.sh
```

## **License**
Please check out the license of each subset we use in our work.

| Dataset Name 	 | License Type   	               |
|----------------|--------------------------------|
| NQ        	    | Apache License 2.0           	 |
| HotpotQA    	  | CC BY-SA 4.0 License           |


## **Citation**

Please cite our paper if you use our data, model or code.

```
@article{longrag,
  title={LongRAG: Enhancing Retrieval-Augmented Generation with Long-context LLMs},
  author={Ziyan Jiang, Xueguang Ma, Wenhu Chen},
  journal={arXiv preprint arXiv:2406.15319},
  year={2024}
}
```
