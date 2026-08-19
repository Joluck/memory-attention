export CUDA_VISIBLE_DEVICES=2
python -m evals.harness --model hf \
    --model_args pretrained='/home/rwkv/jl/flame/exp/mha_340M',max_length=2048,dtype=float16 \
    --tasks wikitext,lambada_openai,piqa,hellaswag,winogrande,arc_easy,arc_challenge,openbookqa \
    --batch_size 32 \
    --num_fewshot 0 \
    --device cuda \
    --show_config \
    --metadata='{"max_seq_lengths":[1024,2048,4096]}'

# wikitext,lambada_openai,piqa,hellaswag,winogrande,arc_easy,arc_challenge,openbookqa
# niah_single_1,niah_single_2,niah_single_3
# triviaqa,swde,drop,nq_open,squadv2,fda
# /mnt/afs/L202500155/ads-cli cp /mnt/afs/L202500155/flame/exp/mla s3://019C28C7F6A0795992B55D984122B177:019C28C7F6A07948B22E12E04512C81D@l202500155.aoss.cn-sh-01b.sensecoreapi-oss.cn/mla
# /home/rwkv/ads-cli cp s3://019C28C7F6A0795992B55D984122B177:019C28C7F6A07948B22E12E04512C81D@l202500155.aoss.cn-sh-01b.sensecoreapi-oss.cn/mla /home/rwkv/jl/models/mla
# |    Tasks     |Version|Filter|n-shot|    Metric     |   | Value |   |Stderr|
# |--------------|------:|------|-----:|---------------|---|------:|---|------|
# |arc_challenge |      1|none  |     0|acc            |↑  | 0.2423|±  |0.0125|
# |              |       |none  |     0|acc_norm       |↑  | 0.2833|±  |0.0132|
# |arc_easy      |      1|none  |     0|acc            |↑  | 0.5606|±  |0.0102|
# |              |       |none  |     0|acc_norm       |↑  | 0.4836|±  |0.0103|
# |hellaswag     |      1|none  |     0|acc            |↑  | 0.3287|±  |0.0047|
# |              |       |none  |     0|acc_norm       |↑  | 0.3847|±  |0.0049|
# |lambada_openai|      1|none  |     0|acc            |↑  | 0.3198|±  |0.0065|
# |              |       |none  |     0|perplexity     |↓  |41.8063|±  |1.6358|
# |openbookqa    |      1|none  |     0|acc            |↑  | 0.2080|±  |0.0182|
# |              |       |none  |     0|acc_norm       |↑  | 0.3200|±  |0.0209|
# |piqa          |      1|none  |     0|acc            |↑  | 0.6654|±  |0.0110|
# |              |       |none  |     0|acc_norm       |↑  | 0.6480|±  |0.0111|
# |wikitext      |      2|none  |     0|bits_per_byte  |↓  | 0.9044|±  |   N/A|
# |              |       |none  |     0|byte_perplexity|↓  | 1.8717|±  |   N/A|
# |              |       |none  |     0|word_perplexity|↓  |28.5619|±  |   N/A|
# |winogrande    |      1|none  |     0|acc            |↑  | 0.5059|±  |0.0141|
