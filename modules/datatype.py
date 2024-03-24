from pydantic import BaseModel
from typing import List, Dict
from numpy import array
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
