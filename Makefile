training_dataset :=
validation_dataset :=
test_dataset :=

python_path :=

model_output_directory :=

batch_size := 1
max_epochs := 30
num_workers := 0

# Fasttext embeddings directory
fasttext_embeddings_directory := 

# Get the full path of the Makefile to set the main path automatically
mkfile_path := $(abspath $(lastword $(MAKEFILE_LIST)))
mkfile_dir := $(dir $(mkfile_path))

main_path := $(mkfile_dir)main.py


train-3-3:
	$(python_path) $(main_path) \
	--embeddings $(fasttext_embeddings_directory)pubmed.fasttext.3-3ngrams.neg5.1e-5_subs.bin \ 
	--training $(training_dataset) \
	--validation $(validation_dataset) \
	--modelOutputFolder $(model_output_directory)3-3ngrams/
	--batchSize $(batch_size) \
	--maxEpochs $(max_epochs) \
	--numWorkers $(num_workers)

train-3-4:
	$(python_path) $(main_path) \
	--embeddings $(fasttext_embeddings_directory)pubmed.fasttext.3-4ngrams.neg5.1e-5_subs.bin \ 
	--training $(training_dataset) \
	--validation $(validation_dataset) \
	--modelOutputFolder $(model_output_directory)3-4ngrams/
	--batchSize $(batch_size) \
	--maxEpochs $(max_epochs) \
	--numWorkers $(num_workers)

train-3-5:
	$(python_path) $(main_path) \
	--embeddings $(fasttext_embeddings_directory)pubmed.fasttext.3-5ngrams.neg5.1e-5_subs.bin \ 
	--training $(training_dataset) \
	--validation $(validation_dataset) \
	--modelOutputFolder $(model_output_directory)3-5ngrams/
	--batchSize $(batch_size) \
	--maxEpochs $(max_epochs) \
	--numWorkers $(num_workers)

train-3-6:
	$(python_path) $(main_path) \
	--embeddings $(fasttext_embeddings_directory)pubmed.fasttext.3-6ngrams.neg5.1e-5_subs.bin \ 
	--training $(training_dataset) \
	--validation $(validation_dataset) \
	--modelOutputFolder $(model_output_directory)3-6ngrams/
	--batchSize $(batch_size) \
	--maxEpochs $(max_epochs) \
	--numWorkers $(num_workers)

train-4-4:
	$(python_path) $(main_path) \
	--embeddings $(fasttext_embeddings_directory)pubmed.fasttext.4-4ngrams.neg5.1e-5_subs.bin \ 
	--training $(training_dataset) \
	--validation $(validation_dataset) \
	--modelOutputFolder $(model_output_directory)4-4ngrams/
	--batchSize $(batch_size) \
	--maxEpochs $(max_epochs) \
	--numWorkers $(num_workers)

train-4-5:
	$(python_path) $(main_path) \
	--embeddings $(fasttext_embeddings_directory)pubmed.fasttext.4-5ngrams.neg5.1e-5_subs.bin \ 
	--training $(training_dataset) \
	--validation $(validation_dataset) \
	--modelOutputFolder $(model_output_directory)4-5ngrams/
	--batchSize $(batch_size) \
	--maxEpochs $(max_epochs) \
	--numWorkers $(num_workers)

train-4-6:
	$(python_path) $(main_path) \
	--embeddings $(fasttext_embeddings_directory)pubmed.fasttext.4-6ngrams.neg5.1e-5_subs.bin \ 
	--training $(training_dataset) \
	--validation $(validation_dataset) \
	--modelOutputFolder $(model_output_directory)4-6ngrams/
	--batchSize $(batch_size) \
	--maxEpochs $(max_epochs) \
	--numWorkers $(num_workers)

train-5-5:
	$(python_path) $(main_path) \
	--embeddings $(fasttext_embeddings_directory)pubmed.fasttext.5-5ngrams.neg5.1e-5_subs.bin \ 
	--training $(training_dataset) \
	--validation $(validation_dataset) \
	--modelOutputFolder $(model_output_directory)5-5ngrams/
	--batchSize $(batch_size) \
	--maxEpochs $(max_epochs) \
	--numWorkers $(num_workers)

train-5-6:
	$(python_path) $(main_path) \
	--embeddings $(fasttext_embeddings_directory)pubmed.fasttext.5-6ngrams.neg5.1e-5_subs.bin \ 
	--training $(training_dataset) \
	--validation $(validation_dataset) \
	--modelOutputFolder $(model_output_directory)5-6ngrams/
	--batchSize $(batch_size) \
	--maxEpochs $(max_epochs) \
	--numWorkers $(num_workers)

train-6-6:
	$(python_path) $(main_path) \
	--embeddings $(fasttext_embeddings_directory)pubmed.fasttext.6-6ngrams.neg5.1e-5_subs.bin \ 
	--training $(training_dataset) \
	--validation $(validation_dataset) \
	--modelOutputFolder $(model_output_directory)6-6ngrams/
	--batchSize $(batch_size) \
	--maxEpochs $(max_epochs) \
	--numWorkers $(num_workers)