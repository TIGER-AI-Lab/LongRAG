import argparse
from utils.load_data_util import load_dict_pickle, save_dict_pickle, load_json_file
from tqdm import tqdm
from pathlib import Path
import os
import heapq
import random

random.seed(2024)
MAX_SIZE = [(0, 10000, 4000)]


def update_degree_dict(degree, corpus_title_set):
    for title in corpus_title_set:
        if title not in degree:
            degree[title] = 0
            abs_adj[title] = set()
    return degree


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_wiki_dir", type=str, default=None, help="Path to the processed Wikipedia dir)")
    parser.add_argument("--mode", type=str, default=None, help="Full wiki/Abstract, full/abs")
    parser.add_argument("--output_dir", type=str, default=None, help="Output dir")

    args = parser.parse_args()

    degree = load_dict_pickle(os.path.join(args.processed_wiki_dir, 'degree.pickle'))
    abs_adj = load_dict_pickle(os.path.join(args.processed_wiki_dir, 'abs_adj.pickle'))
    full_adj = load_dict_pickle(os.path.join(args.processed_wiki_dir, 'full_adj.pickle'))
    doc_size = load_dict_pickle(os.path.join(args.processed_wiki_dir, 'doc_size.pickle'))
    doc_dict = load_dict_pickle(os.path.join(args.processed_wiki_dir, 'doc_dict.pickle'))

    corpus_title_set = set(i for i in doc_size.keys() if doc_size[i] != 0)
    print(f"Num of documents in corpus: {len(corpus_title_set)}")

    if args.mode == 'full':
        other_title_set = set(doc_size.keys()) - corpus_title_set  # full abstract mode use 2M additional nodes
        print(f"Num of additional documents: {len(other_title_set)}")
    elif args.mode == 'abs':
        degree = update_degree_dict(degree, corpus_title_set)  # if a title which is in corpus not in degree, assign 0

    group_size = {}
    group_title = {}  # {id: set{title}}
    group_id = 0
    doc_group_map = {item: -1 for item in corpus_title_set}  # {title: id}

    sorted_degree = dict(sorted(degree.items(), key=lambda item: len(abs_adj[item[0]])))
    for min_d, max_d, max_size in MAX_SIZE:
        filtered_nodes = {k: v for k, v in sorted_degree.items() if min_d < len(abs_adj[k]) <= max_d}
        for node in tqdm(filtered_nodes, desc=f"Grouping from {min_d} to {max_d}"):
            if node not in corpus_title_set:
                continue
            neighbors = {}
            adj = abs_adj[node]
            for i in adj:
                id = doc_group_map.get(i, -1)
                if id != -1:
                    if id in neighbors:
                        neighbors[id] = (neighbors[id][0] + 1, group_size[id])
                    else:
                        neighbors[id] = (1, group_size[id])
            if len(neighbors) == 0:
                group_id += 1
                group_size[group_id] = doc_size[node]
                group_title[group_id] = {node}
                doc_group_map[node] = group_id
                continue
            new_cluster, new_size = {node}, doc_size[node]
            sorted_neighbors = dict(sorted(neighbors.items(), key=lambda item: item[1][1]))
            for id in sorted_neighbors:
                if group_size[id] > max_size - new_size:
                    break
                new_cluster = new_cluster | group_title[id]
                new_size += group_size[id]
            group_id += 1
            group_size[group_id] = new_size
            group_title[group_id] = new_cluster
            for i in new_cluster:
                doc_group_map[i] = group_id

    id_set = set(doc_group_map.values())
    sorted_id = sorted(list(id_set))
    final_group_id = 0
    final_group_title = {}
    final_group_size = {}
    final_doc_group_map = {item: -1 for item in corpus_title_set}
    final_group_text = {}
    for id in tqdm(sorted_id[1:], desc="Final group"):
        final_group_id += 1
        final_group_title[final_group_id] = [(item, degree[item]) for item in group_title[id]]
        final_group_size[final_group_id] = group_size[id]
        final_group_text[final_group_id] = " ".join([doc_dict[item] for item in group_title[id]])
        for i in group_title[id]:
            final_doc_group_map[i] = final_group_id

    print(f"Num of groups: {final_group_id}")
    print(f"Average group size: {sum(final_group_size.values()) / len(final_group_size)}")
    save_dict_pickle(final_doc_group_map, os.path.join(args.output_dir, 'doc_group_map.pickle'))
    save_dict_pickle(final_group_text, os.path.join(args.output_dir, 'group_text.pickle'))
    save_dict_pickle(final_group_title, os.path.join(args.output_dir, 'group_title.pickle'))
    save_dict_pickle(final_group_size, os.path.join(args.output_dir, 'group_size.pickle'))
