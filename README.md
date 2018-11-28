# swagaf

### Like this work, or commonsense reasoning in general? You might be interested in checking out my brand new dataset VCR: Visual Commonsense Reasoning, at [visualcommonsense.com](https://visualcommonsense.com)!

SWAG dataset. More info is at [rowanzellers.com/swag](https://rowanzellers.com/swag).

## Setting up your environment
To create an environment you will need to intall Python 3.1, PyTorch 3.1, and AllenNLP.  These
requirements are listed in `requirements.txt`.

You will also need to set PYTHONPATH to the `swagaf` directory.  You can do this by running the
following command from the `swagaf` folder.

```
export PYTHONPATH=$(pwd)
```

Alternatively, you can build and run the included Dockerfile to create an environment.

```
docker build -t swagaf .
docker run -it swagaf
```

## Common use cases
There is additional documentation in the subfolders.

* `data/` contains the SWAG dataset.
* `swag_baslines/` contains baseline implementations and instructions for how to run them.

Most people will not need to look at `create_swag` or `raw_data` but it's there if you need it!

## Citing

```
@inproceedings{zellers2018swagaf,
    title={SWAG: A Large-Scale Adversarial Dataset for Grounded Commonsense Inference},
    author={Zellers, Rowan and Bisk, Yonatan and Schwartz, Roy and Choi, Yejin},
    booktitle = "Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing (EMNLP)",
    year={2018}
}
```
