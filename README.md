# BioNER

[![Codacy Badge](https://api.codacy.com/project/badge/Grade/df563aa0aa3f49a19e7c8f09df13ae20)](https://app.codacy.com/gh/phil1995/BioNER?utm_source=github.com&utm_medium=referral&utm_content=phil1995/BioNER&utm_campaign=Badge_Grade_Settings)

This repository contains the code for BioNER, an LSTM-based model designed for biomedical named entity recognition (NER).

## Download
We provide the model trained for the following datasets:
| Dataset    | Mirror (Siasky) | Mirror (Mega) |
| ----------- | ----------- | ----------- | 
| MedMentions full | [Download Model](https://siasky.net/EADOnHwSSUVCDfSjIoHCxhkw9KQ3S89dFq_4X8a7QH__Wg) | [Download Model](https://mega.nz/file/axtXRKqC#jWBkkdJJbHjisxZ5XApUsQ0F-wV6DmLCZLyJ8XcJu8o)|
|MedMentions ST21pv | [Download Model](https://siasky.net/EADk7C5dMS6ghnqvgtSAxokLK2lyWgUJW0FnjclWOvj7sQ) | [Download Model](https://mega.nz/file/25lBTIrT#7STIRpqm7tMJ09R9lm4Oa7UKAzst0dLyH3Cl0r19KGs) |
|JNLPBA| [Download Model](https://siasky.net/EAArlhw5cwh0OVX3TX65jZLQWcAxCfJpowJjINAR20_PqA)|[Download Model](https://mega.nz/file/Lx0xSQwT#FTpxQNIOJcm5oq5Uj10xrWQ-elZhef5b5sbPCHs5-6w)|

In addition, the word embeddings trained with fastText on PubMed Baseline 2021 are provided for the following n-gram ranges:
| n-gram  range    | Mirror (Siasky) | Mirror (Mega) | Mirror (Storj)
| ----------- | ----------- | ----------- | ----------- |
| 3-4      | [Download](https://siasky.net/nACvdmBnYm86RAGGvsJLCaIDM2wCDoZK9Yy9_lLp9phgXA) |[Download](https://mega.nz/file/ug9mGTTD#YeFFChChTL5ovZQPA84TH9jHvtdunpj8dJQG4SZ3C2U)|[Download](https://link.eu1.storjshare.io/jwsdq7ymfcnyxnqacofyckxjvyva/bioner%2Fpubmed.fasttext.3-4ngrams.neg5.1e-5_subs.bin)|
| 3-6   | [Download](https://siasky.net/nABUQPit8DTupo4eqidWdWIC9cozk14PiP8eIw2yYNB-BA) | [Download](https://mega.nz/file/ik0hlSyZ#Zjy_whOJtXdt4j8zxC6q9dl7E8lGpXDeCa9lqcw8kTQ) | [Download](https://link.eu1.storjshare.io/jxuer75wl52ijimisfsmwy46lpra/bioner%2Fpubmed.fasttext.3-6ngrams.neg5.1e-5_subs.bin)|

## Installation
Install the dependencies.

```sh
pip install -r requirements.txt
```

As deterministic behaviour is enabled by default, you may need to set a debug environment variable `CUBLAS_WORKSPACE_CONFIG` to prevent RuntimeErrors when using CUDA.
```sh
export CUBLAS_WORKSPACE_CONFIG=:4096:8
```

## Usage
### Dataset Preprocessing
BioNER expects a dataset in the CoNLL-2003 format.
We used the tool [bconv](https://github.com/lfurrer/bconv) for preprocessing the MedMentions dataset.

### Training
You can either use the provided `Makefile` to train the BioNER model or execute `train_bioner.py` directly.
Makefile:
Don't forget to fill in the empty fields in the `Makefile` before the first start.
```sh
make train-bioner ngrams=3-4
```

### Annotation
You can annotate a CoNLL-2003 dataset in the following way:
```sh
python annotate_dataset.py \
--embeddings \ # path to the word embeddings file 
--dataset \ # path to the CoNLL-2003 dataset
--outputFile \ # path to the output file for storing the annotated dataset
--model # path to the trained BioNER model
```
Furthermore, you can add the flag `--enableExportCoNLL` to export an additional file at the same location at the same parent folder as the `outputFile`, which can be used for the evaluation with the original `conlleval.pl` perl script ([source](https://www.clips.uantwerpen.be/conll2003/ner/)).
