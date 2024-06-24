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
In traditional RAG framework, the basic retrieval units are normally short. Such a design forces the 
retriever to search over a large corpus to find the "needle" unit. In contrast, the readers only need 
to extract answers from the short retrieved units. Such an imbalanced heavy retriever and light reader
design can lead to sub-optimal performance. We propose a new framework LongRAG, consisting of a 
"long retriever" and a "long reader". Our framework use a 4K-token retrieval unit, which is 30x longer than before. 
Our study offers insights into the future roadmap for combining RAG with long-context LLMs.

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
This is an optional step. You can use our processed corpus directly. We have released two versions of the retrieval corpus for 
NQ and HotpotQA on Hugging Face.
```python
from datasets import load_dataset
corpus_nq = load_dataset("TIGER-Lab/LongRAG", "nq_corpus")
corpus_hotpotqa = load_dataset("TIGER-Lab/LongRAG", "hotpot_qa_corpus")
```

If you are still interested in how we craft the corpus, you can start reading here.

***Wikipedia raw data clean:***
We first clean Wikipedia raw data by following the standard process. We use [WikiExtractor](https://github.com/attardi/wikiextractor).
This is a widely-used Python script that extracts and cleans text from a Wikipedia database backup dump. Please ensure you use the required 
Python environment. A sample script is:
```bash
sh scripts/extract_and_clean_wiki_dump.sh
```

***Preprocess Wikipedia data***
We are planning to release this data on huggingface too, stay tuned!
After cleaning the Wikipedia raw data, run the following script to gather more information.
```bash
sh scripts/process_wiki_page.sh
```
+ ``dir_path``: The directory path of the cleaned Wikipedia dump, which is the output of the previous step.
+ ``output_path_dir``: The output directory will contain several pickle files, each representing a dictionary for the Wikipedia page. 
``degree.pickle``: The key is the Wikipedia page title, and the value is the number of hyperlinks.
``abs_adj.pickle``: The key is the Wikipedia page title, and the value is the linked page in the abstract paragraph.
``full_adj.pickle``: The key is the Wikipedia page title, and the value is the linked page in the entire page.
``doc_size.pickle``: The key is the Wikipedia page title, and the value is the number of tokens on that page.
``doc_dict.pickle``: The key is the Wikipedia page title, and the value is the text of the page.


***Retrieval Corpus:*** By grouping multiple related documents, we can construct long 
retrieval units with more than 4K tokens. This design could also significantly reduce 
the corpus size (number of retrieval units in the corpus). Then, the retriever’s task 
becomes much easier. Additionally, the long retrieval unit will also improve the 
information completeness to avoid ambiguity or confusion.

```bash
sh scripts/group_documents.sh
```
+ ``processed_wiki_dir``: The output directory of the above step.
+ ``mode``: ``abs`` is for HotpotQA corpus, ``full`` is for NQ corpus.
+ ``output_dir``: The output directory, The output directory will contain several pickle files, each representing a 
dictionary for the retrieval corpus. The most important one is ``group_text.pickle``, which maps the corpus ID to the 
corpus text. For more details, please refer to our released corpus on Hugging Face.








## **Long Retriever**
We leverage open-sourced dense retrieval toolkit, [Tevatron](https://github.com/texttron/tevatron). For all our retrieval experiments. 
The base embedding model we used is bge-large-en-v1.5. We have provided a sample script; make sure to update the parameters with your 
own dataset local path. Additionally, our script uses 4 GPUs to encode the corpus for time saving; please update this based on your own 
use case.
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
