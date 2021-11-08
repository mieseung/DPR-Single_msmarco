import faiss
import pickle
import pandas as pd
import numpy as np
from typing import Dict, List, Union, Optional, Tuple
from search.encoder import QueryEncoder, DenseSearchResult, PRFDenseSearchResult
from transformers.file_utils import requires_backends

from pyserini.search import SimpleSearcher, Document
from pyserini.dsearch import BinaryDenseSearcher, TctColBertQueryEncoder, QueryEncoder, \
    DprQueryEncoder, BprQueryEncoder, DkrrDprQueryEncoder, AnceQueryEncoder, AutoQueryEncoder
from pyserini.util import (download_encoded_queries, download_prebuilt_index,
                           get_dense_indexes_info, get_sparse_index)


class SimpleDenseSearcher:
    """Simple Searcher for dense representation

    Parameters
    ----------
    index_dir : str
        Path to faiss index directory.
    """

    def __init__(self, index_dir: str, query_encoder: Union[QueryEncoder, str],
                 prebuilt_index_name: Optional[str] = None):
        requires_backends(self, "faiss")
        if isinstance(query_encoder, QueryEncoder):
            self.query_encoder = query_encoder
        else:
            self.query_encoder = self._init_encoder_from_str(query_encoder)

        self.index, self.docids = self.load_index(index_dir)
        self.dimension = self.index.d
        self.num_docs = self.index.ntotal

        assert self.docids is None or self.num_docs == len(self.docids)
        if prebuilt_index_name:
            sparse_index = get_sparse_index(prebuilt_index_name)
            self.ssearcher = SimpleSearcher.from_prebuilt_index(sparse_index)

    @classmethod
    def from_prebuilt_index(cls, prebuilt_index_name: str, query_encoder: QueryEncoder):
        """Build a searcher from a pre-built index; download the index if necessary.

        Parameters
        ----------
        query_encoder: QueryEncoder
            the query encoder, which has `encode` method that convert query text to embedding
        prebuilt_index_name : str
            Prebuilt index name.

        Returns
        -------
        SimpleDenseSearcher
            Searcher built from the prebuilt faiss index.
        """
        print(
            f'Attempting to initialize pre-built index {prebuilt_index_name}.')
        try:
            index_dir = download_prebuilt_index(prebuilt_index_name)
        except ValueError as e:
            print(str(e))
            return None

        print(f'Initializing {prebuilt_index_name}...')
        return cls(index_dir, query_encoder, prebuilt_index_name)

    @staticmethod
    def list_prebuilt_indexes():
        """Display information about available prebuilt indexes."""
        get_dense_indexes_info()

    def search(self, query: Union[str, np.ndarray], k: int = 10, threads: int = 1, return_vector: bool = False) \
            -> Union[List[DenseSearchResult], Tuple[np.ndarray, List[PRFDenseSearchResult]]]:
        """Search the collection.

        Parameters
        ----------
        query : Union[str, np.ndarray]
            query text or query embeddings
        k : int
            Number of hits to return.
        threads : int
            Maximum number of threads to use for intra-query search.
        return_vector : bool
            Return the results with vectors
        Returns
        -------
        Union[List[DenseSearchResult], Tuple[np.ndarray, List[PRFDenseSearchResult]]]
            Either returns a list of search results.
            Or returns the query vector with the list of PRF dense search results with vectors.
        """
        if isinstance(query, str):
            emb_q = self.query_encoder.encode(query)
            assert len(emb_q) == self.dimension
            emb_q = emb_q.reshape((1, len(emb_q)))
        else:
            emb_q = query
        faiss.omp_set_num_threads(threads)
        if return_vector:
            distances, indexes, vectors = self.index.search_and_reconstruct(
                emb_q, k)
            vectors = vectors[0]
            distances = distances.flat
            indexes = indexes.flat
            return emb_q, [PRFDenseSearchResult(self.docids[idx], score, vector)
                           for score, idx, vector in zip(distances, indexes, vectors) if idx != -1]
        else:
            distances, indexes = self.index.search(emb_q, k)
            distances = distances.flat
            indexes = indexes.flat
            return [DenseSearchResult(self.docids[idx], score)
                    for score, idx in zip(distances, indexes) if idx != -1]

    def batch_search(self, queries: Union[List[str], np.ndarray], q_ids: List[str], k: int = 10,
                     threads: int = 1, return_vector: bool = False) \
            -> Union[Dict[str, List[DenseSearchResult]], Tuple[np.ndarray, Dict[str, List[PRFDenseSearchResult]]]]:
        """

        Parameters
        ----------
        queries : Union[List[str], np.ndarray]
            List of query texts or list of query embeddings
        q_ids : List[str]
            List of corresponding query ids.
        k : int
            Number of hits to return.
        threads : int
            Maximum number of threads to use.
        return_vector : bool
            Return the results with vectors

        Returns
        -------
        Union[Dict[str, List[DenseSearchResult]], Tuple[np.ndarray, Dict[str, List[PRFDenseSearchResult]]]]
            Either returns a dictionary holding the search results, with the query ids as keys and the
            corresponding lists of search results as the values.
            Or returns a tuple with ndarray of query vectors and a dictionary of PRF Dense Search Results with vectors
        """
        if isinstance(queries, np.ndarray):
            q_embs = queries
        else:
            q_embs = np.array([self.query_encoder.encode(q) for q in queries])
            n, m = q_embs.shape
            assert m == self.dimension
        faiss.omp_set_num_threads(threads)
        if return_vector:
            D, I, V = self.index.search_and_reconstruct(q_embs, k)
            return q_embs, {key: [PRFDenseSearchResult(self.docids[idx], score, vector)
                                  for score, idx, vector in zip(distances, indexes, vectors) if idx != -1]
                            for key, distances, indexes, vectors in zip(q_ids, D, I, V)}
        else:
            D, I = self.index.search(q_embs, k)
            return {key: [DenseSearchResult(self.docids[idx], score)
                          for score, idx in zip(distances, indexes) if idx != -1]
                    for key, distances, indexes in zip(q_ids, D, I)}

    def load_index(self, index_dir: str):
        df = pd.DataFrame()
        for i in range(0, 89):  # need to change to 89
            with open(f'../encoded_passages/passage_embedding_{i}.pkl', 'rb') as f:
                data = pickle.load(f)
                df = pd.concat([df, data])
                f.close()

        emb_np = np.array([emb.T for emb in list(df["embedding"])])

        index = faiss.IndexFlatL2(emb_np.shape[1])
        index.add(emb_np)
        print("The total index is ", index.ntotal, "\n")

        docids = list(df["id"])
        return index, docids

    def doc(self, docid: Union[str, int]) -> Optional[Document]:
        """Return the :class:`Document` corresponding to ``docid``. Since dense indexes don't store documents
        but sparse indexes do, route over to corresponding sparse index (according to prebuilt_index_info.py)
        and use its doc API 

        Parameters
        ----------
        docid : Union[str, int]
            Overloaded ``docid``: either an external collection ``docid`` (``str``) or an internal Lucene ``docid``
            (``int``).

        Returns
        -------
        Document
            :class:`Document` corresponding to the ``docid``.
        """
        return self.ssearcher.doc(docid) if self.ssearcher else None

    @staticmethod
    def _init_encoder_from_str(encoder):
        encoder = encoder.lower()
        if 'dpr' in encoder:
            return DprQueryEncoder(encoder_dir=encoder)
        elif 'tct_colbert' in encoder:
            return TctColBertQueryEncoder(encoder_dir=encoder)
        elif 'ance' in encoder:
            return AnceQueryEncoder(encoder_dir=encoder)
        elif 'sentence' in encoder:
            return AutoQueryEncoder(encoder_dir=encoder, pooling='mean', l2_norm=True)
        else:
            return AutoQueryEncoder(encoder_dir=encoder)

    @staticmethod
    def load_docids(docid_path: str) -> List[str]:
        id_f = open(docid_path, 'r')
        docids = [line.rstrip() for line in id_f.readlines()]
        id_f.close()
        return docids
