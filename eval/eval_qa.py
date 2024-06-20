import argparse
import json
from itertools import islice
from tqdm import tqdm
from utils.bedrock_inference import BedrockInference
from utils.gpt_inference import GPTInference
from utils.gemini_inference import GeminiInference
from utils.claude_inference import ClaudeInference
from utils.eval_util import single_ans_em, has_correct_answer
from utils.load_data_util import load_dict_pickle, load_json_file
import time
import tiktoken
import random


MAX_SIZE = 64000


def get_context(retrieval_result, unit_level="psg", num_unit=1):
    context = []
    retrieve_titles = []
    if unit_level == "psg":
        for i in range(num_unit):
            title, text = retrieval_result[i]["title"], retrieval_result[i]["psg_text"]
            retrieve_titles.append('"' + title + '"')
            context.append(f"Title: {title}\nText:{text}\n")
        return ", ".join(retrieve_titles), "\n".join(context)
    elif unit_level == "doc":
        retrieve_group_id = []
        retrieve_titles = []
        retrieve_size = 0
        for item in retrieval_result:
            if len(retrieve_group_id) == num_unit:
                break
            title = item["title"]
            group_id = doc_group_map[title]
            if group_id in retrieve_group_id:
                continue
            else:
                retrieve_group_id.append(group_id)
                retrieve_titles.append('"' + title + '"')
            retrieve_size += len(enc.encode(chunk_dict[group_id]))
            if retrieve_size > MAX_SIZE:
                break
        context = [group_dict[id] for id in retrieve_group_id]
        return ", ".join(retrieve_titles), " ".join(context)
    else:
        retrieve_group_id = []
        retrieve_titles = []
        for item in retrieval_result:
            if len(retrieve_group_id) == num_unit:
                break
            title = item["title"]
            group_id = doc_group_map[title]
            if group_id in retrieve_group_id or group_id == -1:
                continue
            else:
                retrieve_group_id.append(group_id)
                retrieve_titles += ([i[0] for i in group_titles[group_id]])
        top_1_title = retrieval_result[0]["title"]
        if top_1_title in retrieve_titles:
            retrieve_titles.remove(top_1_title)
        retrieve_titles.insert(0, top_1_title)
        context = [f"Title: {title}\nText:{doc_dict[title]}\n" for title in retrieve_titles]
        retrieve_titles = ['"' + title + '"' for title in retrieve_titles]
        # context = [group_dict[id] for id in retrieve_group_id]
        return ", ".join(retrieve_titles), "\n".join(context)


def generate_prompt(question, context=None):
    if context:
        prompt = f"Context: {context}"
    else:
        prompt = f"Question: {question}"
    return prompt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_file_path", type=str, default=None, help="Path to QA dataset (.json file)")
    parser.add_argument("--unit_level", type=str, default=None, help="Retreive unit level")
    parser.add_argument("--num_unit", type=int, default=None, help="Num of unit")
    parser.add_argument("--output_file_path", type=str, default=None, help="Output file")
    parser.add_argument("--doc_group_map_path", type=str, default=None)
    parser.add_argument("--group_dict_path", type=str, default=None)
    parser.add_argument("--group_titles_path", type=str, default=None)
    parser.add_argument("--doc_dict_path", type=str, default=None)

    args = parser.parse_args()

    enc = tiktoken.get_encoding("cl100k_base")

    test_data = load_json_file(args.test_file_path)
    output_file = open(args.output_file_path, 'w')

    if args.unit_level == "doc":
        doc_group_map = load_dict_pickle(args.doc_group_map_path)
        group_dict = load_dict_pickle(args.group_dict_path)
    elif args.unit_level == "group doc":
        doc_group_map = load_dict_pickle(args.doc_group_map_path)
        group_titles = load_dict_pickle(args.group_titles_path)
        doc_dict = load_dict_pickle(args.doc_dict_path)

    # llm_inference = GeminiInference()
    llm_inference = GPTInference()
    substring_match = 0
    exact_match = 0
    tt = 0
    context_sizes = []
    start_time = time.time()

    for item in tqdm(test_data, desc="Evaluating QA"):
        question, answers = item["question"], item["answers"]
        retrieval_result = item["psgs"]
        titles, context = get_context(retrieval_result, unit_level=args.unit_level, num_unit=args.num_unit)
        context_size = len(enc.encode(context))
        context_sizes.append(context_size)
        try:
            long_ans, short_ans = llm_inference.predict_hqa(context, question, titles)
            # pred = llm_inference.predict_close_book(question)
        except:
            pred = ""
        is_exact_match = single_ans_em(short_ans, answers)
        is_substring_match = has_correct_answer(long_ans, answers)
        # is_retrieval = (item["sp"][0] in titles) and (item["sp"][1] in titles)
        is_retrieval = has_correct_answer(context, answers)

        output = {
            "query_id": item["query_id"],
            "question": question,
            "answers": answers,
            # "pred": pred,
            "long_ans": long_ans,
            "short_ans": short_ans,
            "is_exact_match": is_exact_match,
            "is_substring_match": is_substring_match,
            "is_retrieval": is_retrieval,
        }
        print(output)
        tt += 1
        exact_match += is_exact_match
        substring_match += is_substring_match
        if tt % 10 == 0:
            print(f"Substring match: {substring_match / tt}")
            print(f"Exact match: {exact_match / tt}")
        json_string = json.dumps(output)
        output_file.write(json_string + "\n")

    end_time = time.time()
    print(end_time - start_time)
    print(f"Context size: {sum(context_sizes) / len(context_sizes)}")
