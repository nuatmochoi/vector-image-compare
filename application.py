from pickle import dump, load
from typing import List
from numpy import array
from pathlib import Path
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from modules.datatype import (
    DogOrCat,
    Embeddings,
    EmbeddingsText,
    EmbeddingsImage,
    EmbeddingsWrapper,
    EmbeddingsDataStruct,
)
from modules.embeddings import create_embeddings, create_image_embeddings


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
