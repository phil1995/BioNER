training_dataset :=
validation_dataset :=
test_dataset :=

# Path to venv/bin/python or set to python
python_path :=

# Directory were we create the folders for the models - should include a trailing slash
model_output_directory :=

# Fasttext embeddings directory - should include a trailing slash
fasttext_embeddings_directory :=

batch_size := 64
learning_rate := 0.0005
# Max. epochs a model gets trained - all models get trained via early stopping with a 10 epoch threshold
max_epochs := 300
num_workers := 0

additional_bilstm_layers := 1

# Below are internal variables - do not change

# Get the full path of the Makefile to set the main path automatically
mkfile_path := $(abspath $(lastword $(MAKEFILE_LIST)))
mkfile_dir := $(dir $(mkfile_path))

main_path := $(mkfile_dir)main.py
datexis_path := $(mkfile_dir)datexis.py
parameter_optim_path := $(mkfile_dir)parameter_optimization.py
bioner_path := $(mkfile_dir)train_bioner.py

train-bioner:
	# Pass the ngrams parameter, e.g. make train-bioner ngrams=3-4
	mkdir -p $(model_output_directory)BioNER/$(ngrams)ngrams/batch_size=$(batch_size)_lr=$(learning_rate)/tensorboard_logs  && \
	$(python_path) $(bioner_path) \
	--embeddingsRoot $(fasttext_embeddings_directory) \
	--training $(training_dataset) \
	--validation $(validation_dataset) \
	--modelOutputFolder $(model_output_directory)BioNER/$(ngrams)ngrams/batch_size=$(batch_size)_lr=$(learning_rate)/ \
	--batchSize $(batch_size) \
	--maxEpochs $(max_epochs) \
	--numWorkers $(num_workers) \
	--learningRate $(learning_rate) \
	--trainingsLogFile $(model_output_directory)BioNER/$(ngrams)ngrams/batch_size=$(batch_size)_lr=$(learning_rate)/training.log \
	--tensorboardLogDirectory $(model_output_directory)BioNER/$(ngrams)ngrams/batch_size=$(batch_size)_lr=$(learning_rate)/tensorboard_logs \
	$(args)

train-custom-bioner:
	# Pass the ngrams parameter, e.g. make train-bioner ngrams=3-4
	mkdir -p $(model_output_directory)BioNER/$(ngrams)ngrams/batch_size=$(batch_size)_lr=$(learning_rate)/tensorboard_logs  && \
	$(python_path) $(bioner_path) \
	--embeddings $(fasttext_embeddings_directory)pubmed.fasttext.$(ngrams)ngrams.neg5.1e-5_subs.bin \
	--training $(training_dataset) \
	--validation $(validation_dataset) \
	--modelOutputFolder $(model_output_directory)BioNER/$(ngrams)ngrams/batch_size=$(batch_size)_lr=$(learning_rate)/ \
	--batchSize $(batch_size) \
	--maxEpochs $(max_epochs) \
	--numWorkers $(num_workers) \
	--learningRate $(learning_rate) \
	--trainingsLogFile $(model_output_directory)BioNER/$(ngrams)ngrams/batch_size=$(batch_size)_lr=$(learning_rate)/training.log \
	--tensorboardLogDirectory $(model_output_directory)BioNER/$(ngrams)ngrams/batch_size=$(batch_size)_lr=$(learning_rate)/tensorboard_logs \
	$(args)

train-datexis-ner:
	mkdir -p $(model_output_directory)DATEIXS-NER/batch_size=$(batch_size)_lr=$(learning_rate)/tensorboard_logs && \
	$(python_path) $(datexis_path) \
	--training $(training_dataset) \
	--validation $(validation_dataset) \
	--modelOutputFolder $(model_output_directory)DATEIXS-NER/batch_size=$(batch_size)_lr=$(learning_rate)/ \
	--batchSize $(batch_size) \
	--maxEpochs $(max_epochs) \
	--numWorkers $(num_workers) \
	--learningRate $(learning_rate) \
	--trainingsLogFile $(model_output_directory)DATEIXS-NER/batch_size=$(batch_size)_lr=$(learning_rate)/training.log \
	--tensorboardLogDirectory $(model_output_directory)DATEIXS-NER/batch_size=$(batch_size)_lr=$(learning_rate)/tensorboard_logs \
	$(args)

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
	--model DATEXIS-NER \
	$(args)

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
	--model DATEXIS-NER \
	$(args)

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
	# Pass the ngrams parameter, e.g. make train-custom-DATEXIS ngrams=3-6
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
	--model CustomConfig_DATEXIS-NER \
	--ff1 $(ff) \
	--lstm1 $(lstm) \
	$(args)

train-all-custom-stacked-DATEXIS:
	$(MAKE) train-custom-stacked-DATEXIS ngrams=3-3 ff=$(ff) lstm=$(lstm) && \
	$(MAKE) train-custom-stacked-DATEXIS ngrams=3-4 ff=$(ff) lstm=$(lstm) && \
	$(MAKE) train-custom-stacked-DATEXIS ngrams=3-5 ff=$(ff) lstm=$(lstm) && \
	$(MAKE) train-custom-stacked-DATEXIS ngrams=3-6 ff=$(ff) lstm=$(lstm) && \
	$(MAKE) train-custom-stacked-DATEXIS ngrams=4-4 ff=$(ff) lstm=$(lstm) && \
	$(MAKE) train-custom-stacked-DATEXIS ngrams=4-5 ff=$(ff) lstm=$(lstm) && \
	$(MAKE) train-custom-stacked-DATEXIS ngrams=4-6 ff=$(ff) lstm=$(lstm) && \
	$(MAKE) train-custom-stacked-DATEXIS ngrams=5-5 ff=$(ff) lstm=$(lstm) && \
	$(MAKE) train-custom-stacked-DATEXIS ngrams=5-6 ff=$(ff) lstm=$(lstm) && \
	$(MAKE) train-custom-stacked-DATEXIS ngrams=6-6 ff=$(ff) lstm=$(lstm)

train-custom-stacked-DATEXIS:
	# Pass the ngrams parameter, e.g. make train ngrams=3-6
	mkdir -p $(model_output_directory)stacked_DATEXIS_NER_ADAM/$(ngrams)ngrams/batch_size=$(batch_size)_lr=$(learning_rate)_ff_size=$(ff)_lstm_size=$(lstm)_additional_bilstm_layers=$(additional_bilstm_layers)_dropout=$(dropout)/tensorboard_logs  && \
	$(python_path) $(main_path) \
	--embeddings $(fasttext_embeddings_directory)pubmed.fasttext.$(ngrams)ngrams.neg5.1e-5_subs.bin \
	--training $(training_dataset) \
	--validation $(validation_dataset) \
	--modelOutputFolder $(model_output_directory)stacked_DATEXIS_NER_ADAM/$(ngrams)ngrams/batch_size=$(batch_size)_lr=$(learning_rate)_ff_size=$(ff)_lstm_size=$(lstm)_additional_bilstm_layers=$(additional_bilstm_layers)_dropout=$(dropout)/ \
	--batchSize $(batch_size) \
	--maxEpochs $(max_epochs) \
	--numWorkers $(num_workers) \
	--learningRate $(learning_rate) \
	--trainingsLogFile $(model_output_directory)stacked_DATEXIS_NER_ADAM/$(ngrams)ngrams/batch_size=$(batch_size)_lr=$(learning_rate)_ff_size=$(ff)_lstm_size=$(lstm)_additional_bilstm_layers=$(additional_bilstm_layers)_dropout=$(dropout)/training.log \
	--tensorboardLogDirectory $(model_output_directory)stacked_DATEXIS_NER_ADAM/$(ngrams)ngrams/batch_size=$(batch_size)_lr=$(learning_rate)_ff_size=$(ff)_lstm_size=$(lstm)_additional_bilstm_layers=$(additional_bilstm_layers)_dropout=$(dropout)/tensorboard_logs \
	--model CustomConfig_Stacked-DATEXIS-NER \
	--ff1 $(ff) \
	--lstm1 $(lstm) \
	--additionalBiLSTMLayers $(additional_bilstm_layers) \
	--dropoutProbability $(dropout) \
	$(args)

train-custom-stacked-normalized-DATEXIS:
	# Pass the ngrams parameter, e.g. make train ngrams=3-6
	mkdir -p $(model_output_directory)stacked_normalized_DATEXIS_NER_ADAM/$(ngrams)ngrams/batch_size=$(batch_size)_lr=$(learning_rate)_ff_size=$(ff)_lstm_size=$(lstm)_additional_bilstm_layers=$(additional_bilstm_layers)_dropout=$(dropout)/tensorboard_logs  && \
	$(python_path) $(main_path) \
	--embeddings $(fasttext_embeddings_directory)pubmed.fasttext.$(ngrams)ngrams.neg5.1e-5_subs.bin \
	--training $(training_dataset) \
	--validation $(validation_dataset) \
	--modelOutputFolder $(model_output_directory)stacked_normalized_DATEXIS_NER_ADAM/$(ngrams)ngrams/batch_size=$(batch_size)_lr=$(learning_rate)_ff_size=$(ff)_lstm_size=$(lstm)_additional_bilstm_layers=$(additional_bilstm_layers)_dropout=$(dropout)/ \
	--batchSize $(batch_size) \
	--maxEpochs $(max_epochs) \
	--numWorkers $(num_workers) \
	--learningRate $(learning_rate) \
	--trainingsLogFile $(model_output_directory)stacked_normalized_DATEXIS_NER_ADAM/$(ngrams)ngrams/batch_size=$(batch_size)_lr=$(learning_rate)_ff_size=$(ff)_lstm_size=$(lstm)_additional_bilstm_layers=$(additional_bilstm_layers)_dropout=$(dropout)/training.log \
	--tensorboardLogDirectory $(model_output_directory)stacked_normalized_DATEXIS_NER_ADAM/$(ngrams)ngrams/batch_size=$(batch_size)_lr=$(learning_rate)_ff_size=$(ff)_lstm_size=$(lstm)_additional_bilstm_layers=$(additional_bilstm_layers)_dropout=$(dropout)/tensorboard_logs \
	--model CustomConfig_Stacked-DATEXIS-NER \
	--ff1 $(ff) \
	--lstm1 $(lstm) \
	--additionalBiLSTMLayers $(additional_bilstm_layers) \
	--dropoutProbability $(dropout) \
	--enableBatchNormalization \
	$(args)

parameter-optimization:
	$(python_path) $(parameter_optim_path) \
	--embeddings $(fasttext_embeddings_directory)pubmed.fasttext.$(ngrams)ngrams.neg5.1e-5_subs.bin \
	--training $(training_dataset) \
	--validation $(validation_dataset) \
	--modelOutputFolder $(model_output_directory)stacked_DATEXIS_NER_ADAM/$(ngrams)ngrams/ \
	--additionalBiLSTMLayers $(additional_bilstm_layers) \
	$(args)