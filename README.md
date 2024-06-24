# **LongRAG** 
This repo contains the code for "LongRAG: Enhancing Retrieval-Augmented Generation
with Long-context LLMs". <span style="color: red;">We are still in the process to polish our repo.</span>

<a target="_blank" href="https://arxiv.org/abs/2406.15319">
<img style="height:22pt" src="https://img.shields.io/badge/-Paper-red?style=flat&logo=arxiv"></a>
<a target="_blank" href="https://github.com/TIGER-AI-Lab/LongRAG">
<img style="height:22pt" src="https://img.shields.io/badge/-Code-green?style=flat&logo=github"></a>
<a target="_blank" href="https://tiger-ai-lab.github.io/LongRAG/">
<img style="height:22pt" src="https://img.shields.io/badge/-🌐%20Website-blue?style=flat"></a>
<a target="_blank" href="https://huggingface.co/datasets/TIGER-Lab/LongRAG">
<img style="height:22pt" src="https://img.shields.io/badge/-🤗%20Dataset-red?style=flat"></a>
<a target="_blank" href="">
<img style="height:22pt" src="https://img.shields.io/badge/-Tweet-blue?style=flat&logo=twitter"></a>
<br>


## **Table of Contents**
- [Introduction](#introduction)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Corpus Preparation (Optional)](#corpus)
- [Long Retriever](#long-retriever)
- [Long Reader](#long-reader)
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

## **Quick Start**
Please go to the "Long Reader" section and follow the instructions. This will help you get the final prediction for 100 examples. 
The output will be similar to our sample files in the ``exp/`` directory.

## **Corpus Preparation (Optional)**

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
to handle long context input. (We also plan to test other LLMs capable of handling long contexts in the future.)

The input of the reader is a concatenation of all the long retrieval units from the long retriever.
We have provided the input file in our Huggingface repo.

```bash
mkdir -p exp/
sh scripts/run_eval_qa.sh
```
+ ``test_data_name``: Test set name, ``nq`` (NQ) or ``hotpot_qa`` (HotpotQA).
+ ``test_data_split``: For each test set, there are three splits: ``full``, ``subset_1000``, ``subset_100``. We suggest starting with ``subset_100`` for a 
quick start or debugging and using ``subset_1000`` to obtain relatively stable results.
+ ``output_file_path``: The output file, here it's placed in the ``exp/`` directory.
+ ``reader_model``: The long context reader model we use, currently our code support ``GPT-4o``, ``GPT-4-Turbo``, ``Gemini-1.5-Pro``, ``Claude-3-Opus``.
Please note that you need to update the related API key and API configuration in the code. For example, if you are using the GPT-4 series, you need to 
configure the code in  ``utils/gpt_inference.py``; if you are using the Gemini series, you need to configure the code in  ``utils/gemini_inference.py``. 
We will continue to support more models in the future.

The output file contains one test case per row. The ``short_ans`` field is our final prediction.

```json
{
    "query_id": "383", 
    "question": "how many episodes of touching evil are there", 
    "answers": ["16"], 
    "long_ans": "16 episodes.", 
    "short_ans": "16", 
    "is_exact_match": 1, 
    "is_substring_match": 1, 
    "is_retrieval": 1
}
```
We have provided some sample output files in our `exp/` directory. For example, ``exp/nq_gpt4o_100.json`` contains
the result from the running file:
```bash
python eval/eval_qa.py \
  --test_data_name "nq" \
  --test_data_split "subset_100" \
  --output_file_path "./exp/nq_gpt4o_100.json" \
  --reader_model "GPT-4o"
```
The top-1 retrieval accuracy is 88%, and the exact match rate is 64%.

## **License**
Please check out the license of each subset we use in our work.

| Dataset Name 	 | License Type   	               |
|----------------|--------------------------------|
| NQ        	    | Apache License 2.0           	 |
| HotpotQA    	  | CC BY-SA 4.0 License           |


## **Citation**

Please kindly cite our paper if you find our project is useful:

```
@article{jiang2024longrag
  title={LongRAG: Enhancing Retrieval-Augmented Generation with Long-context LLMs},
  author={Ziyan Jiang, Xueguang Ma, Wenhu Chen},
  journal={arXiv preprint arXiv:2406.15319},
  year={2024},
  url={https://arxiv.org/abs/2406.15319}
}
```
