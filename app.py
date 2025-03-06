import os
import pickle

import openai
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import torch
from sklearn.cluster import KMeans
from transformers import AutoModel, AutoTokenizer


def get_color(value: int) -> str:
    """散布図に使用する色をマッピングする

    Parameters
    ----------
    yeaer : int
        色をマッピングする値(2020~2025)
    Returns
    -------
    str
        色のrgba表現
    """
    if isinstance(value, str):
        return "rgba(255, 0, 0, 1)"  # 文字列の場合は赤色
    # 1990年から2025年を白から青にマッピング
    normalized_value = (value - 2020) / (2025 - 2020)
    blue_intensity = int(255 * normalized_value)
    return f"rgba({255 - blue_intensity}, {255 - blue_intensity}, 255, 1)"


def find_knn(df: pd.DateOffset, n_clusters: int) -> list[str]:
    """KNNによってクラスタリングされたデータの中で最も多いクラスタに属する論文タイトルを抽出する

    Parameters
    ----------
    df : pd.DataFrame
        データ
    n_clusters : int
        クラスタ数

    Returns
    -------
    list[str]
        最も少ないクラスタに属する論文タイトル
    """
    # クラスタリング
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(df[["x", "y"]])
    # 最も多いクラスタに属する論文タイトルを抽出
    df["cluster"] = kmeans.labels_
    cluster_counts = df["cluster"].value_counts()
    min_cluster = cluster_counts.idxmax()
    titles_in_min_cluster = df.loc[
        df["cluster"] == min_cluster, "title"
    ].tolist()
    return titles_in_min_cluster


def extract_research_topic(clusterd_paper_info: list[str]) -> str:
    """論文タイトルから研究トピックを抽出する

    Parameters
    ----------
    clusterd_paper_info : list[str]
        論文タイトルからラベルを抽出

    Returns
    -------
    str
        抽出された研究トピック
    """
    model_name = "Meta-Llama-3.1-8B-Instruct"
    prompt = (
        "Extract three common research topics or keywords from the titles and abstracts of natural language processing research papers. "
        "The output should consist only of the topic keywords."
        "\n\n".join(clusterd_paper_info)
    )

    client = openai.OpenAI(
        api_key=os.environ.get("SAMBANOVA_API_KEY"),
        base_url="https://api.sambanova.ai/v1",
    )

    # Make API call to get model's prediction
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You are a great NLP researcher."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.7,
        top_p=0.9,
    )
    return response.choices[0].message.content


def main():
    # {PCA, UMAP} x {2d, 3d}
    if "all_table" not in st.session_state:
        df_dict: dict[tuple[str, str], pd.DataFrame] = dict()
        for reducer in ["pca", "umap"]:
            for visualize_dim in ["2d", "3d"]:
                df = pd.read_csv(
                    f"./data/embeddings/{reducer}_{visualize_dim}.csv"
                )
                df_dict[(reducer, visualize_dim)] = df
        st.session_state["all_table"] = df_dict

    if "all_reducer" not in st.session_state:
        reducer_dict: dict[tuple[str, str],] = dict()
        for reducer in ["pca", "umap"]:
            for visualize_dim in ["2d", "3d"]:
                with open(
                    f"./model/reducer/{reducer}_{visualize_dim}.pkl",
                    "rb",
                ) as f:
                    model = pickle.load(f)
                reducer_dict[(reducer, visualize_dim)] = model
        st.session_state["all_reducer"] = reducer_dict

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "intfloat/multilingual-e5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)

    st.title("論文埋め込みデモ")

    index = (
        (0 if st.session_state["reducer_name"] == "pca" else 1)
        if "reducer_name" in st.session_state
        else 0
    )
    st.session_state["reducer_name"] = st.selectbox(
        "次元圧縮手法", ["pca", "umap"], index=index
    )
    index = (
        (0 if st.session_state["visualize_dim"] == "2d" else 1)
        if "visualize_dim" in st.session_state
        else 0
    )
    st.session_state["visualize_dim"] = st.selectbox(
        "可視化次元", ["2d", "3d"], index=index
    )

    st.session_state["visualize_data"] = st.session_state["all_table"][
        (st.session_state["reducer_name"], st.session_state["visualize_dim"])
    ]
    st.session_state["reducer"] = st.session_state["all_reducer"][
        (st.session_state["reducer_name"], st.session_state["visualize_dim"])
    ]

    text2 = st.text_input("論文アイデア名")
    embedding_button = st.button("埋め込み")
    if embedding_button:
        with torch.no_grad():
            encoded_input = tokenizer(
                text2, padding=True, truncation=True, return_tensors="pt"
            ).to(device)
            model_output = model(**encoded_input)
            sentence_embedding = torch.mean(
                model_output.last_hidden_state, dim=1
            )
            embedding = st.session_state["reducer"].transform(
                sentence_embedding.cpu().numpy()
            )
            if st.session_state["visualize_dim"] == "2d":
                st.session_state["visualize_data"].loc[
                    len(st.session_state["visualize_data"])
                ] = {
                    "title": text2,
                    "x": embedding[0][0],
                    "y": embedding[0][1],
                    "year": "red",
                }
            elif st.session_state["visualize_dim"] == "3d":
                st.session_state["visualize_data"].loc[
                    len(st.session_state["visualize_data"])
                ] = {
                    "title": text2,
                    "x": embedding[0][0],
                    "y": embedding[0][1],
                    "z": embedding[0][2],
                    "year": "red",
                }
        st.rerun()

    fig = go.Figure()
    color_values = [
        get_color(val) for val in st.session_state["visualize_data"]["year"]
    ]
    if st.session_state["visualize_dim"] == "2d":
        fig.add_trace(
            go.Scatter(
                x=st.session_state["visualize_data"]["x"],
                y=st.session_state["visualize_data"]["y"],
                mode="markers",
                marker=dict(color=color_values),
                hovertemplate="%{text}",
                text=st.session_state["visualize_data"]["title"],
            )
        )
    elif st.session_state["visualize_dim"] == "3d":
        fig.add_trace(
            go.Scatter3d(
                x=st.session_state["visualize_data"]["x"],
                y=st.session_state["visualize_data"]["y"],
                z=st.session_state["visualize_data"]["z"],
                mode="markers",
                marker=dict(color=color_values),
                hovertemplate="%{text}",
                text=st.session_state["visualize_data"]["title"],
            )
        )
    print(type(fig))
    st.plotly_chart(fig)

    title_sampled_button = st.button("人気のあるトピックの論文タイトル抽出")
    if title_sampled_button:
        st.text("論文タイトル")
        sampled_titles = find_knn(st.session_state["visualize_data"], 10)
        for title in sampled_titles:
            st.text(title)
        st.markdown("**トピックキーワード**")
        st.text(extract_research_topic(sampled_titles))
        clear_button = st.button("クリア")
        if clear_button:
            st.session_state["visualize_data"] = st.session_state[
                "visualize_data"
            ][st.session_state["visualize_data"]["year"] != "red"]
            st.rerun()


if __name__ == "__main__":
    main()
