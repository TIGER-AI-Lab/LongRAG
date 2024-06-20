python eval_qa.py \
  --test_file_path "/home/ziyjiang/LongRAG_Data/nq/nq_subset_800.json" \
  --unit_level "group doc" \
  --num_unit 4 \
  --output_file_path "/home/ziyjiang/exp/final_nq/gpt4o_800_ablation.json" \
  --doc_group_map_path "/home/ziyjiang/LongRAG_Data/wiki/doc_group_map.pickle" \
  --group_dict_path "/home/ziyjiang/LongRAG_Data/wiki/group_text.pickle" \
  --group_titles_path "/home/ziyjiang/LongRAG_Data/wiki/group_title.pickle" \
  --doc_dict_path "/home/ziyjiang/LongRAG_Data/wiki/doc_dict.pickle"

python eval_qa.py \
  --test_file_path "/home/ziyjiang/LongRAG_Data/nq/nq_subset_1k.json" \
  --unit_level "psg" \
  --num_unit 100 \
  --output_file_path "/home/ziyjiang/exp/final_nq/gpt4o_close_book.json"

python eval_qa.py \
  --test_file_path "/home/ziyjiang/LongRAG_Data/HotpotQA/hqa_subset_1000.json" \
  --unit_level "group doc" \
  --num_unit 8 \
  --output_file_path "/home/ziyjiang/exp/final_hqa/gemini_group_8.json" \
  --doc_group_map_path "/home/ziyjiang/LongRAG_Data/wiki_2017_abstract_uni/doc_group_map.pickle" \
  --group_dict_path "/home/ziyjiang/LongRAG_Data/wiki_2017_abstract_uni/group_text.pickle" \
  --group_titles_path "/home/ziyjiang/LongRAG_Data/wiki_2017_abstract_uni/group_title.pickle" \
  --doc_dict_path "/home/ziyjiang/LongRAG_Data/wiki_2017_abstract_uni/doc_dict.pickle"
