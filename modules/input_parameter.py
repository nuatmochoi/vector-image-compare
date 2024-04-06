import json
from pydantic import BaseModel
from typing import Optional
from argparse import ArgumentParser
from enum import Enum
from sys import argv


def add_option_flag(parser: ArgumentParser, flag_name: str):
    parser.add_argument(
        flag_name,
        action="store_true",
        default=False,
        required=False,
    )


def add_option_str(
    parser: ArgumentParser, option_name: str, defaults: Optional[str] = None
):
    if defaults is not None:
        parser.add_argument(option_name, type=str, default=defaults, required=False)
    else:
        parser.add_argument(option_name, type=str, required=False)


def add_option_int(
    parser: ArgumentParser, option_name: str, defaults: Optional[int] = None
):
    if defaults is not None:
        parser.add_argument(option_name, type=int, default=defaults, required=False)
    else:
        parser.add_argument(option_name, type=int, required=False)


class ImageType(Enum):
    dog = "dog"
    cat = "cat"


class ProcessName(Enum):
    create = "create"
    init = "init"
    read = "read"
    debug = "debug"


class CohereEmbeddingsType(Enum):
    # "search_document": 検索ユースケースのベクトル データベースに保存された埋め込みに使用されます。
    search_document = "search_document"
    # "search_query": 関連するドキュメントを見つけるためにベクター DB に対して実行される検索クエリの埋め込みに使用されます。
    search_query = "search_query"
    # "classification": テキスト分類子を介して渡される埋め込みに使用されます。
    classification = "classification"
    # "clustering": クラスタリング アルゴリズムを通じて実行される埋め込みに使用されます。
    clustering = "clustering"


class ProcessNameValidate(BaseModel):
    process: ProcessName


def is_json_parameter():
    if len(argv) >= 3:
        try:
            # 受け取ったパラメータがプロセス名であれば、JSONパラメータとして処理しない
            ProcessNameValidate.model_validate({"process": argv[1]})
            return False
        except Exception:
            pass
        try:
            # 受け取ったパラメータがJSONであれば、JSONパラメータとして処理する
            json.loads(argv[1].replace("'", '"'))
            json.loads(argv[2].replace("'", '"'))
            return True
        except Exception:
            pass
    # JSONとして処理しないのなら、FALSEを返す
    return False


parser = ArgumentParser()
parser.add_argument("process", type=str, default="main")
add_option_str(parser, "--embeddings-pickle-file", defaults="embeddings.pkl")
add_option_str(parser, "--index-mask-pickle-file", defaults="index-mask.pkl")
add_option_str(parser, "--create-args-cat-csv-file", defaults="cat_name.csv")
add_option_str(parser, "--create-args-dog-csv-file", defaults="dog_name.csv")
add_option_str(parser, "--image-file")
add_option_str(parser, "--input-text")
add_option_str(parser, "--image-type", defaults="dog")
add_option_flag(parser, "--disable-image-file-cache")
add_option_flag(parser, "--show-criterion-graph")
add_option_flag(parser, "--show-variance-each-index")
add_option_flag(parser, "--init-classification-each-index")
add_option_flag(parser, "--show-classification-result")
add_option_int(parser, "--pca-dims", defaults=2)
add_option_int(parser, "--show-classification-top", defaults=32)
add_option_int(parser, "--embedding-vector-dimensions", defaults=1024)
add_option_str(parser, "--region", defaults="us-east-1")
add_option_str(parser, "--profile", defaults="default")
add_option_str(parser, "--model-id", defaults="amazon.titan-embed-image-v1")
add_option_str(parser, "--dog-target-key", defaults="Dog")
add_option_str(parser, "--cat-target-key", defaults="Cat")
add_option_flag(parser, "--show-original-result")
add_option_str(parser, "--cohere-embeddings-type", defaults="clustering")


class Input(BaseModel):
    process: ProcessName
    embeddings_pickle_file: str
    index_mask_pickle_file: str
    create_args_cat_csv_file: str
    create_args_dog_csv_file: str
    image_file: Optional[str]
    input_text: Optional[str]
    disable_image_file_cache: bool
    image_type: ImageType
    show_criterion_graph: bool
    show_variance_each_index: bool
    init_classification_each_index: bool
    show_classification_result: bool
    pca_dims: int
    show_classification_top: int
    embedding_vector_dimensions: int
    region: str
    profile: str
    model_id: str
    dog_target_key: str
    cat_target_key: str
    show_original_result: bool
    cohere_embeddings_type: CohereEmbeddingsType


def get_python_input() -> Input:
    if not is_json_parameter():
        # ArgParserからパラメータを取得する
        input = Input.model_validate(parser.parse_args().__dict__)
    else:
        # JSONからパラメータを取得する
        argments = {}
        arg_defaults = parser.parse_args(["default"]).__dict__
        input_first = json.loads(argv[2].replace("'", '"'))
        input_defaults = json.loads(argv[1].replace("'", '"'))
        # JSONにデータを格納する
        for key in arg_defaults.keys():
            if key in input_first:
                # 2番目に受け取った引数を最優先とする
                argments[key] = input_first[key]
            elif key in input_defaults:
                # 1番目の引数を上書き可能なデフォルトとして扱う
                argments[key] = input_defaults[key]
            else:
                # どちらの引数でも指定されていないなら、arg_parserから取得する
                argments[key] = arg_defaults[key]
        # JSONから取得する
        input = Input.model_validate(argments)
    # initを実行したのであれば、init_classiication_each_indexのフラグを立てる
    if input.process == ProcessName.init:
        input.process = ProcessName.read
        input.init_classification_each_index = True
        # 画像は分析対象としない
        input.image_file = None
        input.input_text = None
    return input
