import csv
import json
import pickle
from tqdm import tqdm
from utils.wiki_util import _normalize
import pandas as pd


def load_json_file(file_path):
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def load_dpr_wiki(file_path):
    title_set = set()

    with open(file_path, 'r', encoding='utf-8') as tsvfile:
        reader = csv.reader(tsvfile, delimiter='\t')
        headers = next(reader)  # Get the header row

        for row in tqdm(reader, desc="Loading TSV", unit=" rows"):
            # row: id  text  title
            title_set.add(_normalize(row[2]))

    return title_set


def load_tsv_file(file_path):
    data = []
    document_dict = defaultdict(list)
    documents = []

    with open(file_path, 'r', encoding='utf-8') as tsvfile:
        reader = csv.reader(tsvfile, delimiter='\t')
        headers = next(reader)  # Get the header row

        for row in tqdm(reader, desc="Loading TSV", unit=" rows"):
            # row: id  text  title
            data.append(row[1])
            document_dict[row[2]].append(row[1])

    for title, texts in document_dict.items():
        documents.append(" ".join(texts))
    print(len(documents))

    return documents


def save_dict_pickle(dict, path):
    with open(path, 'wb') as handle:
        pickle.dump(dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_dict_pickle(path):
    with open(path, 'rb') as handle:
        dict = pickle.load(handle)
        return dict


def load_retrieval_txt(file_path, n_retrieve=1):
    df = pd.read_csv(file_path, sep='\t', header=None, names=['q_id', 'doc_id', 'score'])
    df = df.groupby('q_id').head(n_retrieve)

    return df
