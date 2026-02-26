# Data Directory
# ==============

This directory contains sample datasets and data download scripts.

## Directory Structure

```
data/
├── README.md
├── .gitkeep
├── sample/              # Sample data for testing
│   └── documents.json
├── scripts/            # Data download scripts
│   └── download_nq.py
└── raw/               # Raw data (gitignored)
```

## Sample Data

The `sample/` directory contains small sample datasets for quick testing.

## Data Download Scripts

Use scripts in `scripts/` to download larger datasets:

```bash
python data/scripts/download_nq.py
```

## Datasets

Common RAG datasets:
- **NQ** (Natural Questions) - Google Search questions
- **TriviaQA** - Trivia questions with evidence
- **HotpotQA** - Multi-hop QA
- **2WikiMultiHopQA** - Multi-hop reasoning

## Notes

- Raw data files are gitignored
- Only keep small sample files in version control
- Use download scripts for full datasets
