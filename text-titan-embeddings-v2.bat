@echo off

REM Create Default Parameter
SET VAL=
SET VAL=%VAL% 'embeddings_pickle_file': 'text2-embeddings-v2.pkl',
SET VAL=%VAL% 'index_mask_pickle_file': 'text2-index-mask-v2.pkl',
SET VAL=%VAL% 'create_args_cat_csv_file': 'yes_data.csv',
SET VAL=%VAL% 'create_args_dog_csv_file': 'no_data.csv',
SET VAL=%VAL% 'model_id': 'amazon.titan-embed-text-v2:0',
SET VAL=%VAL% 'embedding_vector_dimensions': '1024',
SET VAL=%VAL% 'show_original_result': 'true'
SET VAL={%VAL%}

REM Command Sample
REM text-titan-embeddings.bat "{'process': 'read', 'cat_target_key': 'YES Keyword', 'dog_target_key': 'No Keyword'}"
python application.py "%VAL%" %*
