export CUDA_VISIBLE_DEVICES=2
python -m evals.harness --model hf \
    --model_args pretrained='/home/rwkv/jl/flame/exp/egqa_D1024_L24',max_length=2048,dtype=float16 \
    --tasks wikitext,lambada_openai,piqa,hellaswag,winogrande,arc_easy,arc_challenge,openbookqa,niah_single_1,niah_single_2,niah_single_3 \
    --batch_size 32 \
    --num_fewshot 0 \
    --device cuda \
    --show_config \
    --metadata='{"max_seq_lengths":[1024,2048,4096]}'

# wikitext,lambada_openai,piqa,hellaswag,winogrande,arc_easy,arc_challenge,openbookqa
# niah_single_1,niah_single_2,niah_single_3
# triviaqa,swde,drop,nq_open,squadv2,fda
# |    Tasks     |Version|Filter|n-shot|    Metric     |   | Value |   |Stderr|
# |--------------|------:|------|-----:|---------------|---|------:|---|------|
# |arc_challenge |      1|none  |     0|acc            |↑  | 0.2381|±  |0.0124|
# |              |       |none  |     0|acc_norm       |↑  | 0.2611|±  |0.0128|
# |arc_easy      |      1|none  |     0|acc            |↑  | 0.5518|±  |0.0102|
# |              |       |none  |     0|acc_norm       |↑  | 0.4731|±  |0.0102|
# |hellaswag     |      1|none  |     0|acc            |↑  | 0.3207|±  |0.0047|
# |              |       |none  |     0|acc_norm       |↑  | 0.3659|±  |0.0048|
# |lambada_openai|      1|none  |     0|acc            |↑  | 0.3049|±  |0.0064|
# |              |       |none  |     0|perplexity     |↓  |51.1367|±  |2.0380|
# |niah_single_1 |      1|none  |     0|1024           |   | 1.0000|±  |     0|
# |              |       |none  |     0|2048           |   | 0.9540|±  |0.0094|
# |              |       |none  |     0|4096           |↑  | 0.4720|±  |   N/A|
# |niah_single_2 |      1|none  |     0|1024           |   | 0.7100|±  |0.0203|
# |              |       |none  |     0|2048           |   | 0.9220|±  |0.0120|
# |              |       |none  |     0|4096           |↑  | 0.4080|±  |   N/A|
# |niah_single_3 |      1|none  |     0|1024           |   | 0.3740|±  |0.0217|
# |              |       |none  |     0|2048           |   | 0.0820|±  |0.0123|
# |              |       |none  |     0|4096           |↑  | 0.0580|±  |   N/A|
# |openbookqa    |      1|none  |     0|acc            |↑  | 0.2080|±  |0.0182|
# |              |       |none  |     0|acc_norm       |↑  | 0.3180|±  |0.0208|
# |piqa          |      1|none  |     0|acc            |↑  | 0.6572|±  |0.0111|
# |              |       |none  |     0|acc_norm       |↑  | 0.6540|±  |0.0111|
# |wikitext      |      2|none  |     0|bits_per_byte  |↓  | 0.9354|±  |   N/A|
# |              |       |none  |     0|byte_perplexity|↓  | 1.9124|±  |   N/A|
# |              |       |none  |     0|word_perplexity|↓  |32.0450|±  |   N/A|
# |winogrande    |      1|none  |     0|acc            |↑  | 0.5217|±  |0.0140|