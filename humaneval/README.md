# Human Evaluation for Music Anticipatory Infilling Models

## Generating clips for the qualification round

First we select five clips with melodic content from the Lakh MIDI test set: split `f`. Selected clips are stored to the `qualify` directory.
```
python humaneval/melody-select.py $DATAPATH/lmd_full/f/ -o qualify -c 5 -s 1 -v
```

Then we generate accompaniments to these clips. We specify the reference midis (`-d` option) for the retrieval baseline.
```
python humaneval/harmonize.py qualify -r -d $DATAPATH/lmd_full/f/
```

## Generating clips for the prompted completion round

We generate prompted completions using an autoregressive model (or an anticipatory autoresgressive model) checkpoint stored at $MODELPATH.

First, we randomly select 50 prompts and completions from a collection of completions generated using the FIGARO Music Transformer (stored at $FIGARO). Store these prompts at $PROMPTPATH:
```
python humaneval/prompt-select.py $FIGARO -o $PROMPTPATH -c 50 -s 999 -v
```

Generate completions using a model stored at $MODELPATH and store the results to $PROMPTPATH/$OUTPUT$:
```
python humaneval/prompt.py $PROMPTPATH $MODELPATH -o $OUTPUT -c 50 -v
```

Generate completions using an interarrival-time model:
```
python humaneval/prompt-interarrival.py $PROMPTPATH $MODELPATH $OUTPUT -c 50 -v
```

## Generating clips for the accompaniment round

We generate accompaniments using an anticipatory autoregressive model checkpoint stored at $MODELPATH.

First, select 50 clips with melodic content:
```
python humaneval/melody-select.py $DATAPATH/lmd_full/f/ -o harmony -c 50 -v
```

Generate anticipatory accompaniments (`-a` flag):
```
python humaneval/harmonize.py harmony --model $MODELPATH -av -c 50
```

Generate the autoregressive baseline (`-b` flag):
```
python humaneval/harmonize.py harmony --model $MODELPATH -bv -c 50
```

Generate the retrieval baseline (`-r` flag):
```
python humaneval/harmonize.py harmony -d $DATAPATH/lmd_full/f/ -rv -c 50
```
