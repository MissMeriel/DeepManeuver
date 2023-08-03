# Study replication
This subdirectory contains the scripts to reproduce the results tables and figures in the paper study section and the appendix.

## Setup
The study data is available for download from Zenodo.
To download files from Zenodo, you will need the tool [zenodo_get](https://github.com/dvolgyes/zenodo_get).
zenodo_get is [available from pip](https://pypi.org/project/zenodo-get/) or can be installed from source.
First download the study data to the `./data` directory:

```bash
mkdir ../data
pip3 install zenodo_get
zenodo-get -L "<permalink-to-study-data>" ../data
```

## Table and Figure generation

To generate results tables in the paper, run:

```bash
pip install -r study-replication-requirements.txt
python combinatorial.py <result-id>
```

To see a list of available <result-id> arguments, `run python process-results.py -h`

To generate paper figures:
```bash
pip install -r study-replication-requirements.txt
python process-results-for-figures.py figureX
```
