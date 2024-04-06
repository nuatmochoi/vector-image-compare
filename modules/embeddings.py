from base64 import b64encode
from time import sleep
import requests
import json
from typing import List
import boto3
from modules.datatype import Embeddings, EmbeddingsText, EmbeddingsImage
from modules.input_parameter import get_python_input
from pydantic import BaseModel
from pydantic.v1 import BaseSettings
from abc import ABC
from dotenv import load_dotenv

# Provider Name : AWS
PROVIDER_AWS = "AWS"
# Provider Name : OpenAI
PROVIDER_OPENAI = "OpenAI"

# モデルID : Titan Embeddings
TITAN_EMBED_TEXT_V1 = "amazon.titan-embed-text-v1"
# モデルID : Titan Image Embeddings
TITAN_EMBED_IMAGE_TEXT_V1 = "amazon.titan-embed-image-v1"
# モデルID : Cohere Embeddings
COHERE_EMBED_MULTILINGUAL_V3 = "cohere.embed-multilingual-v3"
# モデルID : Ada Embeddings
OPEN_AI_ADA_2_EMBEDDINGS = "text-embedding-ada-002"
# モデルID : OpenAI Embeddings v3
OPEN_AI_EMBEDDINGS_V3 = "text-embedding-3-large"

# 設定を引数から参照する
input_data = get_python_input()


def get_provider(model_id: str):
    """
    プロバイダ名を参照する
    """
    if model_id == TITAN_EMBED_TEXT_V1:
        return PROVIDER_AWS
    if model_id == COHERE_EMBED_MULTILINGUAL_V3:
        return PROVIDER_AWS
    if model_id == TITAN_EMBED_IMAGE_TEXT_V1:
        return PROVIDER_AWS
    return PROVIDER_OPENAI


class OpenAIConfig(BaseSettings):
    open_ai_api_key: str


class EmbeddingRequest(ABC):
    model_id: str

    def create_result(self, response_body) -> Embeddings:
        pass

    @property
    def body(self) -> str:
        pass

    @staticmethod
    def create() -> "EmbeddingRequest":
        pass


class MultimodalTextEmbedding(BaseModel, EmbeddingRequest):
    input_text: str
    output_embedding_length: int
    model_id: str

    @staticmethod
    def create(input_text: str):
        return MultimodalTextEmbedding(
            input_text=input_text,
            model_id=input_data.model_id,
            output_embedding_length=input_data.embedding_vector_dimensions,
        )

    @property
    def body(self):
        # 非マルチモーダルの埋め込みモデルなら、embeddingConfigを指定しない
        if self.model_id == TITAN_EMBED_TEXT_V1:
            # 非マルチモーダル、Titan Embeddings
            return json.dumps(
                {
                    # 対象のテキスト
                    "inputText": self.input_text
                }
            )
        if self.model_id == COHERE_EMBED_MULTILINGUAL_V3:
            # 非マルチモーダル、Cohere Embeddings
            return json.dumps(
                {
                    # 対象のテキスト
                    "texts": [self.input_text],
                    # 埋め込みモデルの利用想定
                    "input_type": input_data.cohere_embeddings_type.value,
                    # トークンの最大長を超えたとき、どのように処理をするか
                    # None -> 何もしない
                    "truncate": "NONE",
                }
            )
        if self.model_id == TITAN_EMBED_IMAGE_TEXT_V1:
            # マルチモーダルの埋め込みモデルを設定する
            return json.dumps(
                {
                    # 対象のテキスト
                    "inputText": self.input_text,
                    # 出力する埋め込みベクトルの次元数
                    "embeddingConfig": {
                        "outputEmbeddingLength": self.output_embedding_length
                    },
                }
            )
        if self.model_id in [OPEN_AI_EMBEDDINGS_V3, OPEN_AI_ADA_2_EMBEDDINGS]:
            # 非マルチモーダル、OpenAI Embeddings
            return json.dumps({"input": self.input_text, "model": self.model_id})
        return ""

    def create_result(self, response_body):
        if self.model_id == COHERE_EMBED_MULTILINGUAL_V3:
            return EmbeddingsText(
                embedding=response_body.get("embeddings")[0], input_text=self.input_text
            )
        if self.model_id in [OPEN_AI_EMBEDDINGS_V3, OPEN_AI_ADA_2_EMBEDDINGS]:
            return EmbeddingsText(
                embedding=response_body.get("data")[0].get("embedding"),
                input_text=self.input_text,
            )
        return EmbeddingsText(
            embedding=response_body.get("embedding"), input_text=self.input_text
        )


class MultimodalImageEmbedding(BaseModel, EmbeddingRequest):
    image_tag: str
    image_name: str
    output_embedding_length: int
    model_id: str

    @staticmethod
    def create(image_tag: str, image_name: str):
        return MultimodalImageEmbedding(
            image_tag=image_tag,
            image_name=image_name,
            model_id=input_data.model_id,
            output_embedding_length=input_data.embedding_vector_dimensions,
        )

    @property
    def body(self):
        with open(self.image_name, "rb") as fp:
            input_image = b64encode(fp.read()).decode("utf-8")

        return json.dumps(
            {
                "inputImage": input_image,
                "embeddingConfig": {
                    "outputEmbeddingLength": self.output_embedding_length
                },
            }
        )

    def create_result(self, response_body):
        return EmbeddingsImage(
            embedding=response_body.get("embedding"),
            input_text=self.image_tag,
            image_name=self.image_name,
        )


def call_api(request: EmbeddingRequest) -> Embeddings:
    if input_data.request_delay != 0:
        # Request Delayが設定されている場合、指定秒数待機する
        # ※RPM制限がAPIがあるため、その場合は--request-delayで待機をかけてください
        sleep(float(input_data.request_delay))
    if get_provider(request.model_id) == PROVIDER_AWS:
        # BedrockのAPIを呼び出す
        bedrock = boto3.Session(
            profile_name=input_data.profile, region_name=input_data.region
        ).client(service_name="bedrock-runtime")

        accept = "application/json"
        content_type = "application/json"

        response = bedrock.invoke_model(
            body=request.body,
            modelId=request.model_id,
            accept=accept,
            contentType=content_type,
        )

        response_body = json.loads(response.get("body").read())
        return request.create_result(response_body)
    elif get_provider(request.model_id) == PROVIDER_OPENAI:
        # 環境変数を.envファイルから読み込む
        load_dotenv()
        # 環境変数を変数に格納する
        open_ai_config = OpenAIConfig()
        # OpenAIのAPIを呼び出す
        response = requests.post(
            url="https://api.openai.com/v1/embeddings",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {open_ai_config.open_ai_api_key}",
            },
            data=request.body,
        )
        response_body = response.json()
        return request.create_result(response_body)


def create_embeddings(input_text: str) -> Embeddings:
    """
    Create Embeddings from Input text
    """
    print(f"START : {input_text}")
    return call_api(MultimodalTextEmbedding.create(input_text))


def create_image_embeddings(image_tag: str, image_name: str) -> Embeddings:
    """
    Create Embeddings from Input text
    """
    print(f"START : {image_name}")
    return call_api(MultimodalImageEmbedding.create(image_tag, image_name))


def text_list_convert_to_embeddings(name_list: List[str]) -> List[Embeddings]:
    """
    テキストの一覧を、一括でベクトル表現に変換する
    name_list: テキストの一覧
    """
    return [create_embeddings(input_text) for input_text in name_list]
