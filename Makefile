training_dataset :=
validation_dataset :=
test_dataset :=

python_path :=

model_output_directory :=

batch_size := 1
learning_rate := 0.001
max_epochs := 30
num_workers := 0

# Fasttext embeddings directory
fasttext_embeddings_directory := 

# Get the full path of the Makefile to set the main path automatically
mkfile_path := $(abspath $(lastword $(MAKEFILE_LIST)))
mkfile_dir := $(dir $(mkfile_path))

main_path := $(mkfile_dir)main.py

train-all: 
	$(MAKE) train ngrams=3-3 && \
	$(MAKE) train ngrams=3-4 && \
	$(MAKE) train ngrams=3-5 && \
	$(MAKE) train ngrams=3-6 && \
	$(MAKE) train ngrams=4-4 && \
	$(MAKE) train ngrams=4-5 && \
	$(MAKE) train ngrams=4-6 && \
	$(MAKE) train ngrams=5-5 && \
	$(MAKE) train ngrams=5-6 && \
	$(MAKE) train ngrams=6-6

train:
	# Pass the ngrams parameter, e.g. make train ngrams=3-6
	mkdir $(model_output_directory)$(ngrams)ngrams/batch_size=$(batch_size)_lr=$(learning_rate) \
	$(model_output_directory)$(ngrams)ngrams/batch_size=$(batch_size)_lr=$(learning_rate)/tensorboard_logs \
	$(python_path) $(main_path) \
	--embeddings $(fasttext_embeddings_directory)pubmed.fasttext.$(ngrams)ngrams.neg5.1e-5_subs.bin \
	--training $(training_dataset) \
	--validation $(validation_dataset) \
	--modelOutputFolder $(model_output_directory)$(ngrams)ngrams/batch_size=$(batch_size)_lr=$(learning_rate)/ \
	--batchSize $(batch_size) \
	--maxEpochs $(max_epochs) \
	--numWorkers $(num_workers) \
	--trainingsLogFile $(model_output_directory)$(ngrams)ngrams/batch_size=$(batch_size)_lr=$(learning_rate)/training.log \
	--tensorboardLogDirectory $(model_output_directory)$(ngrams)ngrams/batch_size=$(batch_size)_lr=$(learning_rate)/tensorboard_logs