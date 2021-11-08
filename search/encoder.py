import os
from dataclasses import dataclass
from typing import Dict, List, Union, Optional, Tuple

import numpy as np
import pandas as pd

from pyserini.util import download_encoded_queries

class QueryEncoder:
    def __init__(self, encoded_query_dir: str = None):
        self.has_model = False
        self.has_encoded_query = False
        if encoded_query_dir:
            self.embedding = self._load_embeddings(encoded_query_dir)
            self.has_encoded_query = True

    def encode(self, query: str):
        return self.embedding[query]

    @classmethod
    def load_encoded_queries(cls, encoded_query_name: str):
        print(f'Attempting to initialize pre-encoded queries {encoded_query_name}.')
        try:
            query_dir = download_encoded_queries(encoded_query_name)
        except ValueError as e:
            print(str(e))
            return None

        print(f'Initializing {encoded_query_name}...')
        return cls(encoded_query_dir=query_dir)

    @staticmethod
    def _load_embeddings(encoded_query_dir):
        df = pd.read_pickle(os.path.join(encoded_query_dir, 'embedding.pkl'))
        return dict(zip(df['text'].tolist(), df['embedding'].tolist()))

@dataclass
class DenseSearchResult:
    docid: str
    score: float


@dataclass
class PRFDenseSearchResult:
    docid: str
    score: float
    vectors: [float]