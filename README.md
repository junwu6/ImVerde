# ImVerde
An implementation for "ImVerde: Vertex-Diminished Random Walk for Learning Imbalanced Network Representation" (Big Data'18). [[Paper]](https://ieeexplore.ieee.org/document/8622603) [[arXiv]](https://arxiv.org/pdf/1804.09222.pdf)

## Environment Requirements
The code is adapted from [[Revisiting Semi-Supervised Learning with Graph Embeddings]](https://github.com/kimiyoung/planetoid). It has been tested under Python 3.6.5. The required packages are listed as follows:
* numpy == 1.17.3
* scipy == 1.3.1
* sklearn ==0.21.3
* Lasagne == 0.2.dev1
* Theano == 1.0.4

## Data sets
We used three public data sets in our experiments: [Cora, Citeseer, Pubmed](https://github.com/kimiyoung/planetoid/tree/master/data).

## Run the Codes
```
python main.py
```

## Acknowledgement
This is the latest source code of **ImVerde** for BigData2018. If you find that it is helpful for your research, please consider to cite our paper:

```
@inproceedings{wu2018imverde,
  title={ImVerde: Vertex-diminished random walk for learning imbalanced network representation},
  author={Wu, Jun and He, Jingrui and Liu, Yongming},
  booktitle={2018 IEEE International Conference on Big Data (Big Data)},
  pages={871--880},
  year={2018},
  organization={IEEE}
}
```
