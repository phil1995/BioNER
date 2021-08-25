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
	mkdir -p $(model_output_directory)$(ngrams)ngrams/batch_size=$(batch_size)_lr=$(learning_rate)/tensorboard_logs  && \
	$(python_path) $(main_path) \
	--embeddings $(fasttext_embeddings_directory)pubmed.fasttext.$(ngrams)ngrams.neg5.1e-5_subs.bin \
	--training $(training_dataset) \
	--validation $(validation_dataset) \
	--modelOutputFolder $(model_output_directory)$(ngrams)ngrams/batch_size=$(batch_size)_lr=$(learning_rate)/ \
	--batchSize $(batch_size) \
	--maxEpochs $(max_epochs) \
	--numWorkers $(num_workers) \
	--learningRate $(learning_rate) \
	--trainingsLogFile $(model_output_directory)$(ngrams)ngrams/batch_size=$(batch_size)_lr=$(learning_rate)/training.log \
	--tensorboardLogDirectory $(model_output_directory)$(ngrams)ngrams/batch_size=$(batch_size)_lr=$(learning_rate)/tensorboard_logs

train-all-original:
	$(MAKE) train-original ngrams=3-3 && \
	$(MAKE) train-original ngrams=3-4 && \
	$(MAKE) train-original ngrams=3-5 && \
	$(MAKE) train-original ngrams=3-6 && \
	$(MAKE) train-original ngrams=4-4 && \
	$(MAKE) train-original ngrams=4-5 && \
	$(MAKE) train-original ngrams=4-6 && \
	$(MAKE) train-original ngrams=5-5 && \
	$(MAKE) train-original ngrams=5-6 && \
	$(MAKE) train-original ngrams=6-6

train-original:
	# Pass the ngrams parameter, e.g. make train ngrams=3-6
	mkdir -p $(model_output_directory)original_DATEXIS_NER/$(ngrams)ngrams/batch_size=$(batch_size)_lr=$(learning_rate)/tensorboard_logs  && \
	$(python_path) $(main_path) \
	--embeddings $(fasttext_embeddings_directory)pubmed.fasttext.$(ngrams)ngrams.neg5.1e-5_subs.bin \
	--training $(training_dataset) \
	--validation $(validation_dataset) \
	--modelOutputFolder $(model_output_directory)original_DATEXIS_NER/$(ngrams)ngrams/batch_size=$(batch_size)_lr=$(learning_rate)/ \
	--batchSize $(batch_size) \
	--maxEpochs $(max_epochs) \
	--numWorkers $(num_workers) \
	--learningRate $(learning_rate) \
	--trainingsLogFile $(model_output_directory)original_DATEXIS_NER/$(ngrams)ngrams/batch_size=$(batch_size)_lr=$(learning_rate)/training.log \
	--tensorboardLogDirectory $(model_output_directory)original_DATEXIS_NER/$(ngrams)ngrams/batch_size=$(batch_size)_lr=$(learning_rate)/tensorboard_logs \
	--model DATEXIS-NER

train-all-original-adam:
	$(MAKE) train-original-adam ngrams=3-3 && \
	$(MAKE) train-original-adam ngrams=3-4 && \
	$(MAKE) train-original-adam ngrams=3-5 && \
	$(MAKE) train-original-adam ngrams=3-6 && \
	$(MAKE) train-original-adam ngrams=4-4 && \
	$(MAKE) train-original-adam ngrams=4-5 && \
	$(MAKE) train-original-adam ngrams=4-6 && \
	$(MAKE) train-original-adam ngrams=5-5 && \
	$(MAKE) train-original-adam ngrams=5-6 && \
	$(MAKE) train-original-adam ngrams=6-6

train-original-adam:
	# Pass the ngrams parameter, e.g. make train ngrams=3-6
	mkdir -p $(model_output_directory)original_DATEXIS_NER_ADAM/$(ngrams)ngrams/batch_size=$(batch_size)_lr=$(learning_rate)/tensorboard_logs  && \
	$(python_path) $(main_path) \
	--embeddings $(fasttext_embeddings_directory)pubmed.fasttext.$(ngrams)ngrams.neg5.1e-5_subs.bin \
	--training $(training_dataset) \
	--validation $(validation_dataset) \
	--modelOutputFolder $(model_output_directory)original_DATEXIS_NER_ADAM/$(ngrams)ngrams/batch_size=$(batch_size)_lr=$(learning_rate)/ \
	--batchSize $(batch_size) \
	--maxEpochs $(max_epochs) \
	--numWorkers $(num_workers) \
	--learningRate $(learning_rate) \
	--trainingsLogFile $(model_output_directory)original_DATEXIS_NER_ADAM/$(ngrams)ngrams/batch_size=$(batch_size)_lr=$(learning_rate)/training.log \
	--tensorboardLogDirectory $(model_output_directory)original_DATEXIS_NER_ADAM/$(ngrams)ngrams/batch_size=$(batch_size)_lr=$(learning_rate)/tensorboard_logs \
	--model DATEXIS-NER

train-all-custom-DATEXIS:
	$(MAKE) train-custom-DATEXIS ngrams=3-3 ff=$(ff) lstm=$(lstm) && \
	$(MAKE) train-custom-DATEXIS ngrams=3-4 ff=$(ff) lstm=$(lstm) && \
	$(MAKE) train-custom-DATEXIS ngrams=3-5 ff=$(ff) lstm=$(lstm) && \
	$(MAKE) train-custom-DATEXIS ngrams=3-6 ff=$(ff) lstm=$(lstm) && \
	$(MAKE) train-custom-DATEXIS ngrams=4-4 ff=$(ff) lstm=$(lstm) && \
	$(MAKE) train-custom-DATEXIS ngrams=4-5 ff=$(ff) lstm=$(lstm) && \
	$(MAKE) train-custom-DATEXIS ngrams=4-6 ff=$(ff) lstm=$(lstm) && \
	$(MAKE) train-custom-DATEXIS ngrams=5-5 ff=$(ff) lstm=$(lstm) && \
	$(MAKE) train-custom-DATEXIS ngrams=5-6 ff=$(ff) lstm=$(lstm) && \
	$(MAKE) train-custom-DATEXIS ngrams=6-6 ff=$(ff) lstm=$(lstm)
	
train-custom-DATEXIS:
	# Pass the ngrams parameter, e.g. make train ngrams=3-6
	mkdir -p $(model_output_directory)original_DATEXIS_NER_ADAM/$(ngrams)ngrams/batch_size=$(batch_size)_lr=$(learning_rate)_ff_size=$(ff)_lstm_size=$(lstm)/tensorboard_logs  && \
	$(python_path) $(main_path) \
	--embeddings $(fasttext_embeddings_directory)pubmed.fasttext.$(ngrams)ngrams.neg5.1e-5_subs.bin \
	--training $(training_dataset) \
	--validation $(validation_dataset) \
	--modelOutputFolder $(model_output_directory)original_DATEXIS_NER_ADAM/$(ngrams)ngrams/batch_size=$(batch_size)_lr=$(learning_rate)_ff_size=$(ff)_lstm_size=$(lstm)/ \
	--batchSize $(batch_size) \
	--maxEpochs $(max_epochs) \
	--numWorkers $(num_workers) \
	--learningRate $(learning_rate) \
	--trainingsLogFile $(model_output_directory)original_DATEXIS_NER_ADAM/$(ngrams)ngrams/batch_size=$(batch_size)_lr=$(learning_rate)_ff_size=$(ff)_lstm_size=$(lstm)/training.log \
	--tensorboardLogDirectory $(model_output_directory)original_DATEXIS_NER_ADAM/$(ngrams)ngrams/batch_size=$(batch_size)_lr=$(learning_rate)_ff_size=$(ff)_lstm_size=$(lstm)/tensorboard_logs \
	--model CustomConfig_DATEXIS-NER