@echo off

REM Create Default Parameter
SET VAL=
SET VAL=%VAL% 'embeddings_pickle_file': 'cohere-embeddings-search_query.pkl',
SET VAL=%VAL% 'index_mask_pickle_file': 'cohere-index-mask-search_query.pkl',
SET VAL=%VAL% 'create_args_cat_csv_file': 'yes_data.csv',
SET VAL=%VAL% 'create_args_dog_csv_file': 'no_data.csv',
SET VAL=%VAL% 'model_id': 'cohere.embed-multilingual-v3',
SET VAL=%VAL% 'embedding_vector_dimensions': '1024',
SET VAL=%VAL% 'cohere_embeddings_type': 'search_query',
SET VAL=%VAL% 'show_original_result': 'true'
SET VAL={%VAL%}

REM Command Sample
REM text-cohere-embeddings-search-query.bat "{'process': 'read', 'cat_target_key': 'YES Keyword', 'dog_target_key': 'No Keyword'}"
python application.py "%VAL%" %*
