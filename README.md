# DPR-Single_msmarco
 Evaluate Natual Questions with DPR-Single model from [pyserini](https://github.com/castorini/pyserini) by using [msmarco dataset](https://github.com/microsoft/MSMARCO-Passage-Ranking)

## dependency
```
pip install -r requirements.txt
pip install faiss-gpu
```
- If you stuck in `faiss` module error, I recommend to uninstall by `pip uninstall faiss-gpu` and reinstall it by `pip install faiss-gpu`.

## encoded_queries & encoded_passages
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

### How to implement this code
[Google colab link](https://colab.research.google.com/drive/1KchkmXpzCfymWwpWFew1Jddrp1yxPqoh?usp=sharing)
