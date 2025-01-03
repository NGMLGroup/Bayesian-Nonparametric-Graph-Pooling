# {Paper Title} ({Venue} {Year})

[![ICLR](https://img.shields.io/badge/{Venue}-{Year}-blue.svg?)]({Link to paper page})
[![paper](https://custom-icon-badges.demolab.com/badge/paper-pdf-green.svg?logo=file-text&logoSource=feather&logoColor=white)]({Link to the paper})

[![poster](https://custom-icon-badges.demolab.com/badge/poster-pdf-orange.svg?logo=note&logoSource=feather&logoColor=white)]({Link to the poster/presentation})
[![arXiv](https://img.shields.io/badge/arXiv-{Arxiv.ID}-b31b1b.svg?)]({Link to Arixv})

This repository contains the code for the reproducibility of the experiments presented in the paper "{Paper Title}" ({Venue} {Year}). {Paper TL;DR}.

**Authors**: [Author 1]({Author1 webpage}), [Author 2]({Author2 webpage})

---

## ⚡ TL;DR

{Paper description}.

<!-- p align=center>
	<img src="./overview.png" alt="{Image description}"/>
</p -->

---

## 📂 Directory structure

The directory is structured as follows:

```
.
├── config/
│   ├── exp1/
│   └── exp2/
├── datasets/
├── lib/
├── requirements.txt
├── conda_env.yaml
└── experiments/
    ├── exp1.py
    └── exp2.py

```


## 📦 Datasets

All datasets are automatically downloaded and stored in the folder `datasets`.

The datasets used in the experiment are provided by [pyg](). Dataset-1 and Dataset-2 datasets are downloaded from these links:
- [Dataset-1]().
- [Dataset-2]().

### New dataset (optional)

In this paper, we introduce a novel dataset {Name of dataset}.

{Dataset TL;DR}.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.{DOI}.svg)]({Link to dataset repository})


## ⚙️ Configuration files

The `config` directory stores all the configuration files used to run the experiment. They are divided into subdirectories according to the experiment they refer to.

## 📝 Requirements

We run all the experiments in `python 3.XX`. To solve all dependencies, we recommend using Anaconda and the provided environment configuration by running the command:

```bash
conda env create -f conda_env.yml
conda activate env_name
```

Alternatively, you can install all the requirements listed in `requirements.txt` with pip:

```bash
pip install -r requirements.txt
```

## 📚 Library

The support code, including the models and the datasets readers, are packed in a python library named `lib`. Should you have to change the paths to the datasets location, you have to edit the `__init__.py` file of the library.


## 🧪 Experiments

The scripts used for the experiments in the paper are in the `experiments` folder.

* `exp1.py` is used to ... . An example of usage is

```bash
python experiments/exp1 --config exp1/config.yaml args
```


## 📖 Bibtex reference

If you find this code useful please consider to cite our paper:

```bibtex
{Bibtex reference}
```
