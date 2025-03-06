from dataclasses import dataclass

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from src.paper import Paper


@dataclass
class PaperEmbedding:
    paper: Paper
    embedding: np.ndarray


def get_paper_embeddings(
    papers: list[Paper],
    model: AutoModel,
    tokenizer: AutoTokenizer,
    add_head_text: str = "",
    batch_size=16,
    device="cpu",
) -> list[PaperEmbedding]:
    """論文埋め込みを取得する

    Parameters
    ----------
    papers : list[Paper]
        論文情報のリスト
    model : AutoModel
        文埋め込みモデル
    tokenizer : AutoTokenizer
        トークナイザ
    add_head_text : str, optional
        入力テキストの先頭につける文字列(e.g. 埋め込みモデルとしてE5を使用する際には"query:"か"passage:"が必要), by default ""
    batch_size : int, optional
        推論させる際のバッチサイズ, by default 16
    device : str, optional
        推論に使うデバイス, by default "cpu"

    Returns
    -------
    list[PaperEmbedding]
        論文埋め込みのリスト
    """

    def _average_pool(last_hidden_states, attention_mask):
        last_hidden = last_hidden_states.masked_fill(
            ~attention_mask[..., None].bool(), 0.0
        )
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    input_texts = [
        " ".join([add_head_text, paper.title, paper.abstract])
        for paper in papers
    ]
    batch_inputs = [
        input_texts[i : i + batch_size]
        for i in range(0, len(input_texts), batch_size)
    ]
    embeddings = []
    for batch_input in tqdm(batch_inputs):
        batch_dict = tokenizer(
            batch_input, padding=True, truncation=True, return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            model_output = model(**batch_dict)
            batch_embeddings = (
                _average_pool(
                    model_output.last_hidden_state,
                    batch_dict["attention_mask"],
                )
                .cpu()
                .numpy()
            )
            for embedding in batch_embeddings:
                embeddings.append(embedding)
    paper_embeddings = [
        PaperEmbedding(paper, embedding)
        for paper, embedding in zip(papers, embeddings)
    ]
    return paper_embeddings
