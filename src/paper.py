import pickle
import random
from dataclasses import dataclass
from itertools import product

import numpy as np
import pandas as pd
import torch
import umap
from acl_anthology import Anthology
from sklearn.decomposition import PCA
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

@dataclass
class Paper:
    title: str
    abstract: str
    year: int
    volume_name: str

def get_papers() -> list[Paper]:
    """2020年〜2023年のACL Anthologyから論文情報を取得する
    会議の対象はACL longのみ  

    Returns
    -------
    list[Paper]
        論文情報のリスト
    """
    papers: list[Paper] = []
    anthology = Anthology.from_repo()

    # 2021〜2023のACL Longを取得
    # ACL Anthologyのフォーマットは{yyyy}.acl-long
    years = list(range(2021, 2024))
    conferences = ["acl"]
    for year, conference in product(years, conferences):
        paper_type = "long" if conference == "acl" else "main"
        volume_name = f"{year}.{conference}-{paper_type}"
        volume = anthology.get_volume(volume_name)
        if volume is None:
            continue
        for paper in volume.papers():
            papers.append(
                Paper(
                    title=str(paper.title),
                    abstract=str(paper.abstract),
                    year=year,
                    volume_name=volume_name,
                )
            )

    # 2020のACL Mainを取得
    # ACL Anthologyのフォーマットは{yyyy}.acl-main
    volume_name = f"2020.acl-main"
    volume = anthology.get_volume(volume_name)
    for paper in volume.papers():
        papers.append(
            Paper(
                title=str(paper.title),
                abstract=str(paper.abstract),
                year=year,
                volume_name=volume,
            )
        )
    return papers