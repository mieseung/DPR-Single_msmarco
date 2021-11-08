# DPR-Single_msmarco
 Evaluate Natual Questions with DPR-Single model from [pyserini](https://github.com/castorini/pyserini) by using [msmarco dataset](https://github.com/microsoft/MSMARCO-Passage-Ranking)

## Modules
```
pip install pyserini
pip install faiss-gpu
```

### encoded_queries & encoded_passages
```
├── encoded_queries
│   └── embeddings.pkl
├── encoded_passages
│   ├── passage_embbeding_0.pkl
│   ├── passage_embbeding_1.pkl
│   ├── ...
│   └── passage_embbeding_88.pkl
├── search
│   ├── encoder.py
│   ├── main.py
│   └── searcher.py
└── README.md
```