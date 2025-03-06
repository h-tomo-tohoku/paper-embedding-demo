import argparse
import os
import pickle
import random
from itertools import product

from transformers import AutoModel, AutoTokenizer

from src.dimension_reduction import paper_embeding_dimension_reduction
from src.embedding import get_paper_embeddings
from src.init_logger import init_logger
from src.paper import Paper, get_papers

random.seed(42)
logger = init_logger("preprocess", "log/preprocess.log")


def main(model_name: str):
    # TODO:モデルを変更できるようにする
    model_name = "intfloat/multilingual-e5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    # 論文情報の取得
    logger.info("Get papers")
    papers: list[Paper] = get_papers()
    papers = random.sample(papers, 500)  # 簡易のためにサンプリング
    logger.info(f"Number of papers: {len(papers)}")

    # 論文埋め込みの取得
    logger.info("Get paper embeddings")
    model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    paper_embeddings = get_paper_embeddings(
        papers=papers,
        model=model,
        tokenizer=tokenizer,
        add_head_text="passage:",
    )
    logger.info(f"Number of paper embeddings: {len(paper_embeddings)}")

    # 次元削減
    logger.info("Dimension reduction")
    reducer_names = ["pca", "umap"]
    reduced_dims = [2, 3]
    for reducer_name, reduced_dim in product(reducer_names, reduced_dims):
        reducer, df = paper_embeding_dimension_reduction(
            paper_embeddings, reducer_name, reduced_dim
        )
        # 結果の保存
        os.makedirs("data", exist_ok=True)
        os.makedirs("data/embeddings", exist_ok=True)
        df.to_csv(
            f"data/embeddings/{reducer_name}_{reduced_dim}d.csv", index=False
        )
        logger.info(f"Saved {reducer_name}_{reduced_dim}d.csv")
        os.makedirs("model", exist_ok=True)
        os.makedirs("model/reducer", exist_ok=True)
        pickle.dump(
            reducer,
            open(f"model/reducer/{reducer_name}_{reduced_dim}d.pkl", "wb"),
        )
        logger.info(f"Saved {reducer_name}_{reduced_dim}d.pkl")
    logger.info("Finished")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    main(args.model_name)
