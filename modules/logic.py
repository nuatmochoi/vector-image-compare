from typing import List
from sklearn.decomposition import PCA
from modules.datatype import (
    DogOrCat,
    EmbeddingsText,
    EmbeddingsWrapper,
    EmbeddingsDataStruct,
)
from modules.input_parameter import get_python_input

# 設定を引数から参照する
input_data = get_python_input()


def print_dog_or_cat(
    data_list: EmbeddingsDataStruct, quiet: bool = False, filter_condition: str = None
):
    """
    犬と猫の分類結果を返却する
    """
    incorrect = 0
    dog_count = 0
    cat_count = 0

    def print_contents(item, required: DogOrCat):
        """
        判定後、ログを出力する
        """
        res = data_list.is_dog_or_cat(item)
        text = item.input_text
        if quiet:
            # ログの出力をしない場合は結果だけを返す
            return res
        if filter_condition is not None:
            # フィルタ条件があるなら、ラベルをフィルタ条件に合わせる
            text = filter_condition
            # フィルタ条件があるなら、フィルタに一致しない場合は結果だけを返す
            if not (filter_condition in item.input_text):
                return res
        # 判定結果を可視化する
        if res == DogOrCat.cat:
            value = input_data.cat_target_key
        else:
            value = input_data.dog_target_key
        # ログ出力して、結果を返す
        if res == required:
            print(
                "{name} : {result} : True".format_map({"name": text, "result": value})
            )
        else:
            print(
                "{name} : {result} : False".format_map({"name": text, "result": value})
            )
        return res

    # 猫の分類結果を取得、期待値と一致しないものをカウントする
    for item in data_list.cat:
        if print_contents(item, DogOrCat.cat) != DogOrCat.cat:
            incorrect += 1
            dog_count += 1
        else:
            cat_count += 1
    # 犬の分類結果を取得、期待値と一致しないものをカウントする
    for item in data_list.dog:
        if print_contents(item, DogOrCat.dog) != DogOrCat.dog:
            incorrect += 1
            cat_count += 1
        else:
            dog_count += 1

    # 結果を返す
    # incorrect : 期待値と一致しない数
    # dog_count : 犬だと判定された数
    # cat_count : 猫だと判定された数
    return (incorrect, dog_count, cat_count)


def reduce_embedding(data_list: EmbeddingsDataStruct, n_components: int = 3):
    """
    埋め込みベクトルの次元数を、PCAを使って削減する
    """
    pca = PCA(n_components=n_components)
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
    """
    特定の次元を選んで、埋め込みベクトルを切り出す
    """
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
