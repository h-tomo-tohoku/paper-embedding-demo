import numpy as np
import pandas as pd
import umap
from sklearn.decomposition import PCA
from src.embedding import PaperEmbedding


def paper_embeding_dimension_reduction(
    paper_embeddings: list[PaperEmbedding],
    dimension_reduction_method: str = "umap",
    reduced_dim: int = 2,
    random_state: int = 42,
) -> tuple[PCA | umap.UMAP, pd.DataFrame]:
    """
    論文埋め込みの情報を次元削減する
    Parameters
    ----------
    paper_embeddings : list[PaperEmbedding]
        論文埋め込みの情報
    dimension_reduction_method : str, optional
        次元削減手法("pca"か"umap"を想定), by default "umap"
    n_components : int, optional
        削減された次元数(2か3を想定), by default 2

    Returns
    -------
    tuple[PCA|umap.UMAP, pd.DataFrame]
        [次元削減に使用したモデル, 次元削減された結果]
        カラム名は["title", "abstract", "year", "volume_name", "x", "y"(, "z")]となる
    """
    if dimension_reduction_method not in ["pca", "umap"]:
        raise ValueError(
            f"Invalid dimension_reduction_method: {dimension_reduction_method}"
        )
    if reduced_dim not in [2, 3]:
        raise ValueError(f"Invalid reduced_dim: {reduced_dim}")
    reducer = (
        PCA(n_components=reduced_dim)
        if dimension_reduction_method == "pca"
        else umap.UMAP(n_components=reduced_dim, random_state=random_state)
    )
    embeddings = np.array([e.embedding for e in paper_embeddings])
    reduced_embeddings = reducer.fit_transform(embeddings)
    reduced_embeddings = reducer.fit_transform(
        [embedding.embedding for embedding in paper_embeddings]
    )
    titles = [embedding.paper.title for embedding in paper_embeddings]
    abstracts = [embedding.paper.abstract for embedding in paper_embeddings]
    years = [embedding.paper.year for embedding in paper_embeddings]
    volumes = [embedding.paper.volume_name for embedding in paper_embeddings]
    xs = [embedding[0] for embedding in reduced_embeddings]
    ys = [embedding[1] for embedding in reduced_embeddings]
    df = pd.DataFrame(
        {
            "title": titles,
            "abstract": abstracts,
            "year": years,
            "volume_name": volumes,
            "x": xs,
            "y": ys,
        }
    )
    if reduced_dim == 3:
        # z軸の情報がある場合はz座標も追加
        zs = [embedding[2] for embedding in reduced_embeddings]
        df["z"] = zs
    return reducer, df
