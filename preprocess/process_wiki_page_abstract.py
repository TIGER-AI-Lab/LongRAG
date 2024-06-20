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
                title = page_data["title"]

                try:
                    abstract = "".join(page_data["text"][1])
                    text = " ".join(["".join(page_data["text"][i]) for i in range(len(page_data["text"]))])
                    abs_hyperlink, full_hyperlink = get_hyperlink(text, abstract)

                    new_page_data = {
                        "title": title,
                        "abs_hyperlink": abs_hyperlink,
                        "full_hyperlink": full_hyperlink,
                    }
                    data.append(new_page_data)
                except:
                    continue

    return data


def get_degree_dict():
    degree = {}
    for item in tqdm(full_adj, desc="Generate degree dict"):
        degree[item] = len(full_adj[item])

    return degree


def get_adjacency():
    abs_adj_uni = {item: set() for item in title_set}
    abs_adj = {item: set() for item in title_set}
    full_adj = {item: set() for item in title_set}

    for item in tqdm(processed_data, desc="Generate abs adjacency dict"):
        title = item["title"]
        if title in title_set:  # in corpus
            for i in item["abs_hyperlink"]:
                if i in title_set:
                    abs_adj[title].add(i)
                    abs_adj[i].add(title)
                    abs_adj_uni[title].add(i)
                elif i.lower() in title_map:
                    abs_adj[title].add(title_map[i.lower()])
                    abs_adj[title_map[i.lower()]].add(title)
                    abs_adj_uni[title].add(title_map[i.lower()])

    for item in tqdm(processed_data, desc="Generate full adjacency dict"):
        title = item["title"]
        if title in title_set:  # in corpus
            for i in item["full_hyperlink"]:
                if i in title_set:
                    full_adj[title].add(i)
                    full_adj[i].add(title)
                elif i.lower() in title_map:
                    full_adj[title].add(title_map[i.lower()])
                    full_adj[title_map[i.lower()]].add(title)

    return abs_adj, full_adj, abs_adj_uni


if __name__ == "__main__":
    enc = tiktoken.get_encoding("cl100k_base")

    dir_path = Path("/home/ziyjiang/LongRAG_Data/wiki_raw_2017/")
    output_path_dir = Path("/home/ziyjiang/LongRAG_Data/wiki_2017_abstract/")
    file_paths = [file_path for file_path in dir_path.rglob('*') if file_path.is_file()]

    util = ProcessWikipedia(func=process_wiki, data=file_paths, n_processes=16)
    processed_data = util.process_data()
    doc_size = load_dict_pickle(os.path.join(output_path_dir, "doc_size.pickle"))
    title_set = set(doc_size.keys())
    title_map = {title.lower(): title for title in title_set}
    abs_adj, full_adj, abs_adj_uni = get_adjacency()
    degree = get_degree_dict()

    save_dict_pickle(degree, os.path.join(output_path_dir, 'degree.pickle'))
    save_dict_pickle(abs_adj, os.path.join(output_path_dir, 'abs_adj.pickle'))
    save_dict_pickle(full_adj, os.path.join(output_path_dir, 'full_adj.pickle'))
    save_dict_pickle(abs_adj_uni, os.path.join(output_path_dir, 'abs_adj_uni.pickle'))

    print(f"Avg Degree: {sum(degree.values()) / len(degree)}")
    breakpoint()
