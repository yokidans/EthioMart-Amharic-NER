# EthioMart Amharic NER 

## Overview
A Named Entity Recognition (NER) system for Ethiopian e-commerce Telegram data, focusing on:
- Product names (`B-PRODUCT`, `I-PRODUCT`)
- Prices (`B-PRICE`, `I-PRICE`) 
- Locations (`B-LOC`, `I-LOC`)

## Workflow Steps

### Task 1: Data Ingestion & Preprocessing
    git checkout -b task1-data-ingestion

 ## 1. Data Collection
- Script: src/data_ingestion/telegram_scraper.py

- Input: 23 Ethiopian e-commerce channels

- Output:
   ###  data/raw/
### ├── channel1_messages.csv
### ├── channel2_messages.csv
### └── all_messages_combined.csv

## 1. Annotation Guidelines
| Amharic Text      | CoNLL Tag    | Description               |
|-------------------|-------------|--------------------------|
| "ስልክ 5000 ብር"    | B-PRODUCT   | First product word        |
| "5000"            | B-PRICE     | Numeric price value      |
| "ብር"             | I-PRICE     | Currency unit            |


