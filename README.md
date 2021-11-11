# BioNER
This repository contains the code for BioNER, an LSTM-based model designed for biomedical named entity recognition (NER).

## Download
We provide the model trained for the following datasets:
- [MedMentions full](https://mega.nz/file/axtXRKqC#jWBkkdJJbHjisxZ5XApUsQ0F-wV6DmLCZLyJ8XcJu8o)
- [MedMentions ST21pv](https://mega.nz/file/25lBTIrT#7STIRpqm7tMJ09R9lm4Oa7UKAzst0dLyH3Cl0r19KGs)
- [JNLPBA](https://mega.nz/file/Lx0xSQwT#FTpxQNIOJcm5oq5Uj10xrWQ-elZhef5b5sbPCHs5-6w)

In addition, the word embeddings trained with fastText on PubMed Baseline 2021 are provided for the following n-gram ranges:
- [3-4](https://mega.nz/file/ug9mGTTD#YeFFChChTL5ovZQPA84TH9jHvtdunpj8dJQG4SZ3C2U) 
- [3-6](https://mega.nz/file/ik0hlSyZ#Zjy_whOJtXdt4j8zxC6q9dl7E8lGpXDeCa9lqcw8kTQ)

## Installation
Install the dependencies.

```sh
pip install -r requirements.txt
```

## Usage
### Dataset Preprocessing
BioNER expects a dataset in the CoNLL-2003 format.
We used the tool [bconv](https://github.com/lfurrer/bconv) for preprocessing the MedMentions dataset.

### Training
You can either use the provided `Makefile` to train the BioNER model or execute the `main.py` directly.
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
Furthermore, you can add the flag `--enableExportCoNLL` to export an additional file at the same location at the same parent folder as the `outputFile`, which can be used for the evaluation with the original `conlleval.pl` script.
