# Study replication
This subdirectory contains the scripts to reproduce the results tables in the paper study section and the appendix.

## Setup
First download the study data to the `./data` directory:

```bash
curl -L "<permalink-to-study-data>"
```

## Table and Figure generation

To generate results tables in the paper, run:

```bash
pip install -r study-replication-requirements.txt
python process-results.py <result-id>
```

To see a list of available <result-id> arguments, `run python process-results.py -h`

To generate paper figures:
```bash
pip install -r study-replication-requirements.txt
python process-results-for-figures.py figureX
```
