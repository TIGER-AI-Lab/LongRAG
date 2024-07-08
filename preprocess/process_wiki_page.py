import argparse
import bz2
from collections import defaultdict
import csv
import itertools
import json
from transformers import AutoTokenizer
from utils.mp_util import MultiprocessingUtil
from utils.load_data_util import load_dpr_wiki, load_dict_pickle, save_dict_pickle
from utils.wiki_util import _normalize, convert_html, get_hyperlink, remove_hyperlink
from pathlib import Path
from tqdm import tqdm
import os
import tiktoken


class ProcessWikipedia(MultiprocessingUtil):
    def combine_results(self, result_chunks):
        result = list(itertools.chain.from_iterable(result_chunks))
        return result


def process_wiki(file_paths):
    data = []
    for file_path in tqdm(file_paths, desc="Processing wiki files"):
        with bz2.open(file_path, "rb") as file:
            for line in file:
                line_decoded = _normalize(line.decode('utf-8'))
                page_data = json.loads(line_decoded)
                title = convert_html(page_data["title"])
                abs_hyperlink, full_hyperlink = get_hyperlink(page_data["text"])

                new_page_data = {"title": title,
                                 "url": page_data["url"],
                                 "text": remove_hyperlink(page_data["text"], abstract=False),
                                 "abs_hyperlink": abs_hyperlink,
                                 "full_hyperlink": full_hyperlink}
                new_page_data["size"] = len(enc.encode(new_page_data["text"]))

                if title in corpus_title_set:
                    new_page_data["in_corpus"] = True
                else:
                    new_page_data["in_corpus"] = False

                data.append(new_page_data)

    return data


def get_degree_dict():
    degree = {}
    for item in tqdm(full_adj, desc="Generate degree dict"):
        degree[item] = len(full_adj[item])

    return degree


def get_adjacency():
    abs_adj = {item: set() for item in title_set}
    full_adj = {item: set() for item in title_set}

    for item in tqdm(processed_data, desc="Generate abs adjacency dict"):
        title = item["title"]
        if title in title_set:  # in corpus
            for i in item["abs_hyperlink"]:
                if i in title_set:
                    abs_adj[title].add(i)
                    abs_adj[i].add(title)
                elif i.lower() in title_map:
                    abs_adj[title].add(title_map[i.lower()])
                    abs_adj[title_map[i.lower()]].add(title)

    for item in tqdm(processed_data, desc="Generate full adjacency dict"):
        title = item["title"]
        if title in title_set:  # in corpus
            for i in item["full_hyperlink"]:
                if i in doc_size:
                    full_adj[title].add(i)
                    full_adj[i].add(title)
                elif i.lower() in title_map:
                    full_adj[title].add(title_map[i.lower()])
                    full_adj[title_map[i.lower()]].add(title)

    return abs_adj, full_adj


def get_doc_size():
    doc_size = {}
    for item in tqdm(processed_data, desc="Generate doc size dict"):
        if item["in_corpus"]:
            doc_size[item["title"]] = item["size"]

    return doc_size


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir_path", type=str, default=None, help="Path to the cleaned Wikipedia dir)")
    parser.add_argument("--output_path_dir", type=str, default=None, help="Output dir")
    parser.add_argument("--corpus_title_path", type=str, default=None, help="Used for filtering the title")

    args = parser.parse_args()

    enc = tiktoken.get_encoding("cl100k_base")
    file_paths = [file_path for file_path in Path(args.dir_path).rglob('*') if file_path.is_file()]
    corpus_title_set = load_dpr_wiki(args.corpus_title_path)
    print(f"Size of Corpus: {len(corpus_title_set)}")

    util = ProcessWikipedia(func=process_wiki, data=file_paths, n_processes=16)
    processed_data = util.process_data()

    doc_size = get_doc_size()
    title_set = set(doc_size.keys())
    title_map = {title.lower(): title for title in title_set}

    abs_adj, full_adj = get_adjacency()
    degree = get_degree_dict()

    save_dict_pickle(degree, os.path.join(args.output_path_dir, 'degree.pickle'))
    save_dict_pickle(abs_adj, os.path.join(args.output_path_dir, 'abs_adj.pickle'))
    save_dict_pickle(full_adj, os.path.join(args.output_path_dir, 'full_adj.pickle'))
    save_dict_pickle(doc_size, os.path.join(args.output_path_dir, 'doc_size.pickle'))
    print(f"Num of All Pages: {len(title_set)}")

    doc_dict = {}
    for item in tqdm(processed_data, desc="Generate doc dict"):
        doc_dict[item["title"]] = item["text"]
    save_dict_pickle(doc_dict, os.path.join(args.output_path_dir, 'doc_dict.pickle'))
