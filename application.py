from pickle import dump, load
from numpy import array
from pathlib import Path
from matplotlib import pyplot as plt
from modules.datatype import (  # noqa: F401
    Embeddings,
    EmbeddingsImage,
    EmbeddingsDataStruct,
    EmbeddingsText,  # pickleが参照するため、importする
    DogOrCat,  # pickleが参照するため、importする
    EmbeddingsWrapper,  # pickleが参照するため、importする
)
from modules.embeddings import (
    create_embeddings,
    create_image_embeddings,
    text_list_convert_to_embeddings,
)
from modules.input_parameter import Input, ProcessName, get_python_input
from modules.logic import crop_embedding, print_dog_or_cat, reduce_embedding
import japanize_matplotlib  # noqa: F401


def read_name_list(file_name: str):
    """
    ファイルから対象データ(犬の品種名、猫の品種名)を読み込む
    file_name: ファイル名
    """
    with open(file_name, "r", encoding="utf-8") as fp:
        name_list = [line.rstrip() for line in fp.readlines() if len(line) > 0]
    return sorted(name_list)


def create_embedding_pickle_data(input_data: Input):
    """
    文字列データをベクトル表現に変換して、pickleファイルに保存する
    """
    dictionary_text = {
        "Positive": input_data.dog_target_key,
        "Negative": input_data.cat_target_key,
    }
    # 対象のデータファイル（テキスト）を読み込む
    cat_list = read_name_list(input_data.create_args_cat_csv_file)
    dog_list = read_name_list(input_data.create_args_dog_csv_file)
    # 対象のテキストをベクトル表現に変換する
    cat_vector_list = text_list_convert_to_embeddings(cat_list)
    dog_vector_list = text_list_convert_to_embeddings(dog_list)
    dog_dictionary_vector = text_list_convert_to_embeddings([dictionary_text["Positive"]])
    cat_dictionary_vector = text_list_convert_to_embeddings([dictionary_text["Negative"]])
    with open(input_data.embeddings_pickle_file, "wb") as file:
        # Pickle形式で保存する
        dump(
            EmbeddingsDataStruct(
                cat=cat_vector_list,
                dog=dog_vector_list,
                cat_dictionary=cat_dictionary_vector[0],
                dog_dictionary=dog_dictionary_vector[0],
            ),
            file,
        )


def visualize_request_with_pickle(input_data: Input, appends: Embeddings | None = None):
    """
    Pickleを利用してベクトル表現を可視化する

    input_data: 入力データ
    appends: 引数<image_file>として入力された画像データのベクトル表現
    """
    # Pickleから全体のベクトル表現を読みこむ
    plt.rc('font', family='Malgun Gothic')
    plt.rcParams['axes.unicode_minus'] = False

    data_list: EmbeddingsDataStruct
    with open(input_data.embeddings_pickle_file, "rb") as file:
        data_list = load(file)

    # もし<image_file>として画像を受け取っているのなら、データを追加する
    if appends is not None:
        if isinstance(appends, EmbeddingsImage):
            if "dog" in appends.input_text:
                # <image_type>がdogであれば、犬データの場所に格納する
                data_list.dog.append(appends)
            else:
                # <image_type>がcatであれば、猫データの場所に格納する
                # ※格納場所は分類結果に影響しないが、猫だけをフィルタリングして可視化するときに使う
                data_list.cat.append(appends)

    if input_data.show_criterion_graph:
        # 辞書データをそのままグラフとして表示する
        print("SHOW CRITERION GRAPH")
        plt.plot(range(1, 51), data_list.dog_dictionary.embedding[:50], color="red")
        plt.plot(range(1, 51), data_list.cat_dictionary.embedding[:50], color="blue")
        plt.show()
        return

    if input_data.show_variance_each_index:
        # 各インデックスごとに分散を集計、表示する
        print("SHOW VARIANCE EACH INDEX")
        variance_list = []
        for i in range(0, input_data.embedding_vector_dimensions):
            embeddings = data_list.transpose_embeddings(i)
            variance_list.append(embeddings.variance)

        plt.plot(
            range(1, input_data.embedding_vector_dimensions + 1),
            variance_list,
            color="blue",
        )
        plt.show()
        return

    if input_data.init_classification_each_index:
        # 各インデックスごとに[犬-猫]のどちらに近いかを集計する
        std_log_buffer = []
        index_mask = []
        incorrect_histgram = []
        reduced_incorrect_histgram = []
        for i in range(0, input_data.embedding_vector_dimensions):
            # それぞれのインデックスを評価する
            reduced = crop_embedding(data_list, index=i)
            # incorrect : 分類の不正解数
            # dog_count : 犬だと判定された数
            # cat_count : 猫だと判定された数
            incorrect, dog_count, cat_count = print_dog_or_cat(reduced, quiet=True)
            # ログ出力用に不正解数を格納する
            incorrect_histgram.append(incorrect)
            # 分類されなかったデータ(全て犬だと判定された、またはその逆)は除外する
            # 分類された(全てのレコードが一方に偏らなかった)インデックスだけを、マスク情報として記録する
            if dog_count != 0 and cat_count != 0:
                std_log_buffer.append(f"{i}\t{incorrect}")
                index_mask.append(i)
                # マスク後の不正解数を格納する
                reduced_incorrect_histgram.append(incorrect)
        # マスク情報を標準出力に出力する
        print("\n".join(std_log_buffer))
        # マスク情報をファイルに出力する
        with open(input_data.index_mask_pickle_file, "wb") as file:
            dump(index_mask, file)

        # グラフとして表示する
        figure = plt.figure()
        ax1 = figure.add_subplot(1, 2, 1)
        ax1.plot(
            range(1, len(incorrect_histgram) + 1),
            incorrect_histgram,
            color="blue",
        )
        ax2 = figure.add_subplot(1, 2, 2)
        ax2.plot(
            range(1, len(reduced_incorrect_histgram) + 1),
            reduced_incorrect_histgram,
            color="blue",
        )
        plt.show()
        return

    # インデックスマスクがあれば、マスクを適用する
    if Path(input_data.index_mask_pickle_file).exists():
        with open(input_data.index_mask_pickle_file, "rb") as file:
            index_mask = load(file)

        # マスクに一致するデータ（分類に利用可能なインデックスのデータ）だけを残す
        data_list = crop_embedding(data_list, index=index_mask)

    # 分析を開始
    if input_data.show_classification_result:
        std_log_buffer = []
        for i in range(1, input_data.show_classification_top + 1):
            reduced = reduce_embedding(data_list, n_components=i)
            incorrect, _, _ = print_dog_or_cat(reduced, quiet=True)
            std_log_buffer.append(f"{i}\t{incorrect}")
        print("\n".join(std_log_buffer))
        return

    # 次元数を削減する
    reduced = reduce_embedding(data_list, n_components=input_data.pca_dims)

    # 犬-猫の分類結果を表示する
    if input_data.image_file is not None:
        # 引数に<image_file>があるなら、対象の画像データの分類結果だけを出力する
        print(f"Reduced Data: dims ({input_data.pca_dims})")
        print_dog_or_cat(reduced, quiet=False, filter_condition="#input")
        print(f"Non Reduced Data: dims ({input_data.embedding_vector_dimensions})")
        print_dog_or_cat(data_list, quiet=False, filter_condition="#input")
    else:
        if input_data.show_original_result:
            print(f"Non Reduced Data: dims ({input_data.embedding_vector_dimensions})")
            res_incorrect, res_dog, res_cat = print_dog_or_cat(data_list, quiet=False)
            print("--------------------------------------------")
            print(
                f"Result : {(1.0 - (float(res_incorrect) / (res_dog + res_cat))) * 100}%"
            )
        else:
            print(f"Reduced Data: dims ({input_data.pca_dims})")
            res_incorrect, res_dog, res_cat = print_dog_or_cat(reduced, quiet=False)
            print("--------------------------------------------")
            print(
                f"Result : {(1.0 - (float(res_incorrect) / (res_dog + res_cat))) * 100}%"
            )

    # 次元数を削減したベクトルデータを取得する
    embeddings_reduced = array([e.value.embedding for e in reduced.entity_list])

    # 各点に対応するテキストをプロットに追加
    for i, entity in enumerate(data_list.entity_list):
        # 犬のデータは赤色、猫のデータは青色で表示する
        color = "blue" if "cat" in entity.key else "red"
        # 対象のテキスト(犬、猫の品種名)を参照する
        text = entity.value.input_text
        # <image_file>で引数を受け取ったのなら、オレンジ色の点で表示する
        if "#input" in text:
            color = "orange"
            text = "[Input Image]"
        # 散布図としてプロットする
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
    # 判例を追加
    plt.legend([input_data.cat_target_key, input_data.dog_target_key])
    # グラフを表示する
    plt.show()


def read_image(actual_tag: str, image_name: str, disabled_cache: bool = False):
    """
    画像を読み込んでベクトル化する

    actual_tag: 画像のタグ
    image_name: 対象になる画像ファイル
    disabled_cache: Trueならキャッシュせず、都度ベクトル表現をAPIから取得する
    """
    # Pickleでキャッシュする
    path = Path(f"{image_name}.pkl")
    if (not path.exists()) or (disabled_cache):
        # もしキャッシュが存在しないのなら、APIを実行してベクトル表現を作成する
        image_embedding = create_image_embeddings(actual_tag, image_name)
        with open(path, "wb") as fp:
            dump(image_embedding, fp)
    else:
        # キャッシュからベクトル表現を読み込む
        with open(path, "rb") as fp:
            image_embedding = load(fp)
    # ベクトル表現を返す
    return image_embedding


if __name__ == "__main__":
    input_data = get_python_input()
    if input_data.process == ProcessName.create:
        # ベクトル表現を作成する
        create_embedding_pickle_data(input_data)
    elif input_data.process == ProcessName.read:
        # ベクトル表現をPickleから読み込んで処理する
        if input_data.image_file is not None:
            # 画像があるなら、画像をベクトル化する
            data = read_image(
                f"{input_data.image_type.value}.#input",
                input_data.image_file,
                disabled_cache=input_data.disable_image_file_cache,
            )
            # ベクトル化した画像を可視化する
            visualize_request_with_pickle(input_data, data)
        elif input_data.input_text is not None:
            # テキストがあるなら、テキストをベクトル化する
            data = create_embeddings(input_data.input_text)
            # ベクトル化したテキストを可視化する
            visualize_request_with_pickle(input_data, data)
        else:
            # データ全体を可視化する
            visualize_request_with_pickle(input_data)
    elif input_data.process == ProcessName.debug:
        # 引数を出力する
        print("DEBUG")
        print(input_data)
