from base64 import b64encode
import json
from typing import List
import boto3
from modules.datatype import Embeddings, EmbeddingsText, EmbeddingsImage
from modules.input_parameter import get_python_input

# 設定を引数から参照する
input_data = get_python_input()


def create_embeddings(input_text: str) -> Embeddings:
    """
    Create Embeddings from Input text
    """
    print(f"START : {input_text}")

    model_id = input_data.model_id
    output_embedding_length = input_data.embedding_vector_dimensions

    # Create request body.
    body = json.dumps(
        {
            "inputText": input_text,
            "embeddingConfig": {"outputEmbeddingLength": output_embedding_length},
        }
    )

    bedrock = boto3.Session(
        profile_name=input_data.profile, region_name=input_data.region
    ).client(service_name="bedrock-runtime")

    accept = "application/json"
    content_type = "application/json"

    response = bedrock.invoke_model(
        body=body, modelId=model_id, accept=accept, contentType=content_type
    )

    response_body = json.loads(response.get("body").read())
    return EmbeddingsText(
        embedding=response_body.get("embedding"), input_text=input_text
    )


def create_image_embeddings(image_tag: str, image_name: str) -> Embeddings:
    """
    Create Embeddings from Input text
    """
    print(f"START : {image_name}")

    model_id = input_data.model_id
    output_embedding_length = input_data.embedding_vector_dimensions

    with open(image_name, "rb") as fp:
        input_image = b64encode(fp.read()).decode("utf-8")

    # Create request body.
    body = json.dumps(
        {
            "inputImage": input_image,
            "embeddingConfig": {"outputEmbeddingLength": output_embedding_length},
        }
    )

    bedrock = boto3.Session(
        profile_name=input_data.profile, region_name=input_data.region
    ).client(service_name="bedrock-runtime")

    accept = "application/json"
    content_type = "application/json"

    response = bedrock.invoke_model(
        body=body, modelId=model_id, accept=accept, contentType=content_type
    )

    response_body = json.loads(response.get("body").read())
    return EmbeddingsImage(
        embedding=response_body.get("embedding"),
        input_text=image_tag,
        image_name=image_name,
    )


def text_list_convert_to_embeddings(name_list: List[str]) -> List[Embeddings]:
    """
    テキストの一覧を、一括でベクトル表現に変換する
    name_list: テキストの一覧
    """
    return [create_embeddings(input_text) for input_text in name_list]
