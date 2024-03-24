import json
import boto3
from pydantic import BaseModel
from pickle import dump, load
from typing import List, Dict
from numpy import array
from base64 import b64encode
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA, FastICA, FactorAnalysis
from sklearn.metrics.pairwise import cosine_similarity
from enum import Enum


class DogOrCat(Enum):
    dog = "dog"
    cat = "cat"


class Embeddings(BaseModel):
    embedding: List[float]

    def find_nearest(self, sampling: Dict[DogOrCat, List[float]]) -> DogOrCat:
        """
        類似度を求める
        """
        result = {}
        current = 999.9
        return_key = DogOrCat.cat
        for key, value in sampling.items():
            # コサイン類似度を取得する
            cos_sim = cosine_similarity(array([self.embedding]), array([value]))[0][0]
            # コサイン距離に変換する
            result[key] = 1.0 - cos_sim
            # コサイン距離の最小を探索。最小値を更新したのなら、更新した最小値を保持する
            if result[key] < current:
                current = result[key]
                return_key = key
        # 最も近いキーを返す
        return return_key

    @property
    def variance(self) -> DogOrCat:
        """
        分散を求める
        """
        return array(self.embedding).var()


class EmbeddingsText(Embeddings):
    input_text: str


class EmbeddingsImage(Embeddings):
    input_text: str
    image_name: str


class EmbeddingsWrapper(BaseModel):
    value: Embeddings
    key: str


class EmbeddingsDataStruct(BaseModel):
    cat: List[Embeddings]
    dog: List[Embeddings]
    cat_dictionary: Embeddings
    dog_dictionary: Embeddings

    def is_dog_or_cat(self, embeddings: Embeddings):
        nearest = embeddings.find_nearest(
            {
                DogOrCat.dog: self.dog_dictionary.embedding,
                DogOrCat.cat: self.cat_dictionary.embedding,
            }
        )
        return nearest

    def transpose_embeddings(self, index: int):
        """
        転置する
        特定のインデックスだけを抜き出した配列を作成する
        """
        result = []
        for _, e in enumerate(self.entity_list):
            result.append(e.value.embedding[index])
        return Embeddings(embedding=result)

    @property
    def entity_list(self):
        result: List[EmbeddingsWrapper] = []
        result.append(
            EmbeddingsWrapper(value=self.cat_dictionary, key="cat_dictionary")
        )
        result.append(
            EmbeddingsWrapper(value=self.dog_dictionary, key="dog_dictionary")
        )
        result.extend([EmbeddingsWrapper(value=cat, key="cat") for cat in self.cat])
        result.extend([EmbeddingsWrapper(value=dog, key="dog") for dog in self.dog])
        return result

    @staticmethod
    def from_reduced_embeddings(values: List[EmbeddingsWrapper]):
        cat = []
        dog = []
        cat_dictionary = None
        dog_dictionary = None
        for value in values:
            if value.key == "cat":
                cat.append(value.value)
            elif value.key == "dog":
                dog.append(value.value)
            elif value.key == "cat_dictionary":
                cat_dictionary = value.value
            elif value.key == "dog_dictionary":
                dog_dictionary = value.value
        return EmbeddingsDataStruct(
            cat=cat,
            dog=dog,
            cat_dictionary=cat_dictionary,
            dog_dictionary=dog_dictionary,
        )


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


def read_name_list(file_name: str):
    with open(file_name, "r", encoding="utf-8") as fp:
        name_list = [line.rstrip() for line in fp.readlines() if len(line) > 0]
    return sorted(name_list)


def main():
    dictionary_text = {
        "dog": "Dog",
        "cat": "Cat",
    }
    cat_list = read_name_list("cat_name.csv")
    dog_list = read_name_list("dog_name.csv")
    cat_vector_list = text_convert_to_embeddings(cat_list)
    dog_vector_list = text_convert_to_embeddings(dog_list)
    dog_dictionary_vector = text_convert_to_embeddings([dictionary_text["dog"]])
    cat_dictionary_vector = text_convert_to_embeddings([dictionary_text["cat"]])
    with open("embeddings.pkl", "wb") as file:
        dump(
            EmbeddingsDataStruct(
                cat=cat_vector_list,
                dog=dog_vector_list,
                cat_dictionary=cat_dictionary_vector[0],
                dog_dictionary=dog_dictionary_vector[0],
            ),
            file,
        )


def text_convert_to_embeddings(name_list: List[str]) -> List[Embeddings]:
    return [create_embeddings(input_text) for input_text in name_list]


def print_dog_or_cat(data_list: EmbeddingsDataStruct, quiet: bool = False):
    incorrect = 0
    dog_count = 0
    cat_count = 0

    def print_contents(item):
        res = data_list.is_dog_or_cat(item)
        if not quiet:
            print(
                "{name} : {result}".format_map(
                    {"name": item.input_text, "result": res.value}
                )
            )
        return res

    for item in data_list.cat:
        if print_contents(item) != DogOrCat.cat:
            incorrect += 1
            dog_count += 1
        else:
            cat_count += 1
    for item in data_list.dog:
        if print_contents(item) != DogOrCat.dog:
            incorrect += 1
            cat_count += 1
        else:
            dog_count += 1

    return (incorrect, dog_count, cat_count)


def reduce_embedding(data_list: EmbeddingsDataStruct, n_components: int = 3):
    if True:
        pca = PCA(n_components=n_components)
    elif False:
        pca = FactorAnalysis(n_components=n_components)
    else:
        pca = FastICA(n_components=n_components, random_state=0)
    embeddings_reduced = pca.fit_transform(
        [e.value.embedding for e in data_list.entity_list]
    )

    return EmbeddingsDataStruct.from_reduced_embeddings(
        [
            EmbeddingsWrapper(
                value=EmbeddingsText(
                    embedding=embeddings_reduced[i].tolist(),
                    input_text=entity.value.input_text,
                ),
                key=entity.key,
            )
            for i, entity in enumerate(data_list.entity_list)
        ]
    )


def crop_embedding(data_list: EmbeddingsDataStruct, index: int | List[int]):
    if isinstance(index, int):
        # 単一の次元を切り出す
        embeddings_reduced = [[e.value.embedding[index]] for e in data_list.entity_list]
    else:
        # 該当する次元を切り出す
        embeddings_reduced = [
            [e.value.embedding[i] for i in index] for e in data_list.entity_list
        ]

    return EmbeddingsDataStruct.from_reduced_embeddings(
        [
            EmbeddingsWrapper(
                value=EmbeddingsText(
                    embedding=embeddings_reduced[i],
                    input_text=entity.value.input_text,
                ),
                key=entity.key,
            )
            for i, entity in enumerate(data_list.entity_list)
        ]
    )


def reads(appends: Embeddings | None = None):
    data_list: EmbeddingsDataStruct
    with open("embeddings.pkl", "rb") as file:
        data_list = load(file)

    contents = []

    if appends is not None:
        if isinstance(appends, EmbeddingsImage):
            if "dog" in appends.input_text:
                data_list.dog.append(appends)
            else:
                data_list.cat.append(appends)

    if False:
        plt.plot(range(1, 51), data_list.dog_dictionary.embedding[:50], color="red")
        plt.plot(range(1, 51), data_list.cat_dictionary.embedding[:50], color="blue")
        plt.show()
        return

    if False:
        icr = []
        for i in range(0, 1024):
            embeddings = data_list.transpose_embeddings(i)
            icr.append(embeddings.variance)

        plt.plot(range(1, 1025), icr, color="blue")
        plt.show()

        return

    if False:
        icr = []
        for i in range(0, 1024):
            reduced = crop_embedding(data_list, index=i)
            incorrect, dog_count, cat_count = print_dog_or_cat(reduced, quiet=True)
            if dog_count != 0 and cat_count != 0:
                contents.append(f"{i}\t{incorrect}")
                icr.append(i)
        print("\n".join(contents))
        with open("index2.pkl", "wb") as file:
            dump(icr, file)

        # plt.plot(range(1, 1025), icr, color="blue")
        # plt.show()

        return

    with open("index2.pkl", "rb") as file:
        icr = load(file)

    data_list = crop_embedding(data_list, index=icr)

    # 分析を開始
    if True:
        for i in range(1, 32):
            reduced = reduce_embedding(data_list, n_components=i)
            incorrect, _, _ = print_dog_or_cat(reduced, quiet=True)
            contents.append(f"{i}\t{incorrect}")
        print("\n".join(contents))

    dims = 3
    reduced = reduce_embedding(data_list, n_components=dims)
    indexes = range(1, dims + 1)

    print_dog_or_cat(reduced, quiet=False)

    embeddings_reduced = array([e.value.embedding for e in reduced.entity_list])

    # 各点に対応するテキストをプロットに追加
    for i, entity in enumerate(data_list.entity_list):
        if True:
            color = "blue" if "cat" in entity.key else "red"
            text = entity.value.input_text
            if "#input" in text:
                color = "orange"
                text = "[Input Image]"
            plt.scatter(
                embeddings_reduced[i, 0],
                embeddings_reduced[i, 1],
                color=color,
            )
            plt.text(
                embeddings_reduced[i, 0],
                embeddings_reduced[i, 1],
                text,
            )
    plt.legend(["Cat", "Dog"])
    plt.show()


def read_image(actual_tag: str, image_name: str):
    from pathlib import Path

    path = Path(f"{image_name}.pkl")
    if not path.exists():
        image_embedding = create_image_embeddings(actual_tag, image_name)
        with open(path, "wb") as fp:
            dump(image_embedding, fp)
    else:
        with open(path, "rb") as fp:
            image_embedding = load(fp)
    return image_embedding


if __name__ == "__main__":
    dog = read_image("cat.#input", "dogear.png")
    # main()
    reads(dog)
