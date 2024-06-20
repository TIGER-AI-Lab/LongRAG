for s in $(seq -f "%02g" 0 4)
do
CUDA_VISIBLE_DEVICES=${s} python -m tevatron.retriever.driver.encode \
  --output_dir=temp \
  --model_name_or_path BAAI/bge-large-en-v1.5 \
  --normalize True \
  --fp16 \
  --per_device_eval_batch_size 128 \
  --passage_max_len 512 \
  --dataset_path "/home/ziyjiang/LongRAG_Data/wiki_2017_abstract/wiki_2017_corpus.json" \
  --dataset_number_of_shards 4 \
  --encode_output_path emb_bge/corpus_emb_${s}.pkl \
  --dataset_shard_index ${s} &
done

python -m tevatron.retriever.driver.encode \
  --output_dir=temp \
  --model_name_or_path BAAI/bge-large-en-v1.5  \
  --normalize True \
  --query_prefix "Represent this sentence for searching relevant passages: " \
  --fp16 \
  --per_device_eval_batch_size 256 \
  --dataset_path "/home/ziyjiang/LongRAG_Data/HotpotQA/hotpot_qa_test.json" \
  --encode_output_path hqa_query.pkl \
  --query_max_len 32 \
  --encode_is_query

python -m tevatron.retriever.driver.search \
  --query_reps hqa_query.pkl \
  --passage_reps "emb_bge/corpus_emb*.pkl" \
  --depth 200 \
  --batch_size -1 \
  --save_text \
  --save_ranking_to hqa_rank_200_new.txt
