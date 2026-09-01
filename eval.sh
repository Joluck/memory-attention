export CUDA_VISIBLE_DEVICES=2

python -m evals.harness --model hf \
    --model_args pretrained='/home/rwkv/jl/flame/exp/eva',max_length=2048,dtype=float16 \
    --tasks wikitext,lambada_openai,piqa,hellaswag,winogrande,arc_easy,arc_challenge,openbookqa \
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

# |    Tasks     |Version|Filter|n-shot|    Metric     |   | Value |   |Stderr|
# |--------------|------:|------|-----:|---------------|---|------:|---|------|
# |arc_challenge |      1|none  |     0|acc            |↑  | 0.2415|±  |0.0125|
# |              |       |none  |     0|acc_norm       |↑  | 0.2619|±  |0.0128|
# |arc_easy      |      1|none  |     0|acc            |↑  | 0.5505|±  |0.0102|
# |              |       |none  |     0|acc_norm       |↑  | 0.4979|±  |0.0103|
# |hellaswag     |      1|none  |     0|acc            |↑  | 0.3228|±  |0.0047|
# |              |       |none  |     0|acc_norm       |↑  | 0.3746|±  |0.0048|
# |lambada_openai|      1|none  |     0|acc            |↑  | 0.3173|±  |0.0065|
# |              |       |none  |     0|perplexity     |↓  |41.9115|±  |1.6411|
# |niah_single_1 |      1|none  |     0|1024           |   | 0.9980|±  |0.0020|
# |              |       |none  |     0|2048           |   | 0.9920|±  |0.0040|
# |              |       |none  |     0|4096           |↑  | 0.5000|±  |   N/A|
# |niah_single_2 |      1|none  |     0|1024           |   | 0.4320|±  |0.0222|
# |              |       |none  |     0|2048           |   | 0.9460|±  |0.0101|
# |              |       |none  |     0|4096           |↑  | 0.4660|±  |   N/A|
# |niah_single_3 |      1|none  |     0|1024           |   | 0.6960|±  |0.0206|
# |              |       |none  |     0|2048           |   | 0.3520|±  |0.0214|
# |              |       |none  |     0|4096           |↑  | 0.0780|±  |   N/A|
# |openbookqa    |      1|none  |     0|acc            |↑  | 0.2140|±  |0.0184|
# |              |       |none  |     0|acc_norm       |↑  | 0.3260|±  |0.0210|
# |piqa          |      1|none  |     0|acc            |↑  | 0.6556|±  |0.0111|
# |              |       |none  |     0|acc_norm       |↑  | 0.6507|±  |0.0111|
# |wikitext      |      2|none  |     0|bits_per_byte  |↓  | 0.9129|±  |   N/A|
# |              |       |none  |     0|byte_perplexity|↓  | 1.8829|±  |   N/A|
# |              |       |none  |     0|word_perplexity|↓  |29.4849|±  |   N/A|
# |winogrande    |      1|none  |     0|acc            |↑  | 0.5138|±  |0.0140|