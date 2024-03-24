from base64 import b64encode
import json
import boto3
from modules.datatype import Embeddings, EmbeddingsText, EmbeddingsImage


def create_embeddings(input_text: str) -> Embeddings:
    """
    Create Embeddings from Input text
    """
    print("STARt")

    model_id = "amazon.titan-embed-image-v1"
    output_embedding_length = 1024

    # Create request body.
    body = json.dumps(
        {
            "inputText": input_text,
            "embeddingConfig": {"outputEmbeddingLength": output_embedding_length},
        }
    )

    bedrock = boto3.client(service_name="bedrock-runtime", region_name="us-east-1")

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
    print("STARt")

    model_id = "amazon.titan-embed-image-v1"
    output_embedding_length = 1024

    with open(image_name, "rb") as fp:
        input_image = b64encode(fp.read()).decode("utf-8")

    # Create request body.
    body = json.dumps(
        {
            "inputImage": input_image,
            "embeddingConfig": {"outputEmbeddingLength": output_embedding_length},
        }
    )

    bedrock = boto3.client(service_name="bedrock-runtime", region_name="us-east-1")

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
