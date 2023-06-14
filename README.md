# Anticipatory Music Transformer

Implementation of the models described in Anticipatory Music Transformer.

by [__John Thickstun__](https://johnthickstun.com/), [__David Hall__](http://dlwh.org/), [__Chris Donahue__](https://chrisdonahue.com/), and [__Percy Liang__](https://cs.stanford.edu/~pliang/).

-------------------------------------------------------------------------------------

This repository provides the code for creating anticipatory training datasets, and for sampling from models trained with anticipation. It does _not_ provide code for training these models: you may use the datasets constructed here as input to your favorite codebase for training autoregressive transformer models.

This project is licensed under the terms of the Apache License, Version 2.0.

Begin by installing the anticipation package (from the root anticipation package directory).

```
pip install .
```

## Software Dependencies

Run the following command to install dependencies.

```
pip install -r requirements.txt
```

## Generating Music with an Anticipatory Music Transformer


## Training an Anticipatory Music Transformer

See the [train](train) directory for instructions on preprocessing the Lakh MIDI dataset and using [Levanter](https://github.com/stanford-crfm/levanter) to train an Anticipatory Music Transformer.

## Reproducing the Human Evaluation Procedure

See the [humaneval](humaneval) directory for instructions on reproducing data used for human evaluations.
