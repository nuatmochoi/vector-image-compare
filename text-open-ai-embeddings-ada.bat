@echo off

REM Create Default Parameter
SET VAL=
SET VAL=%VAL% 'embeddings_pickle_file': 'openai-embeddings-ada.pkl',
SET VAL=%VAL% 'index_mask_pickle_file': 'openai-index-mask-ada.pkl',
SET VAL=%VAL% 'create_args_cat_csv_file': 'yes_data.csv',
SET VAL=%VAL% 'create_args_dog_csv_file': 'no_data.csv',
SET VAL=%VAL% 'model_id': 'text-embedding-ada-002',
SET VAL=%VAL% 'embedding_vector_dimensions': '1536',
SET VAL=%VAL% 'request_delay': '30',
SET VAL=%VAL% 'show_original_result': 'true'
SET VAL={%VAL%}

REM Command Sample
REM text-open-ai-embeddings-ada.bat "{'process': 'read', 'cat_target_key': 'YES Keyword', 'dog_target_key': 'No Keyword'}"
python application.py "%VAL%" %*
