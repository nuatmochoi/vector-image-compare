from pydantic import BaseModel
from typing import Optional
from argparse import ArgumentParser
from enum import Enum


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


def get_python_input() -> Input:
    input = Input.model_validate(parser.parse_args().__dict__)
    # initを実行したのであれば、init_classiication_each_indexのフラグを立てる
    if input.process == ProcessName.init:
        input.process = ProcessName.read
        input.init_classification_each_index = True
        # 画像は分析対象としない
        input.image_file = None
        input.input_text = None
    return input
