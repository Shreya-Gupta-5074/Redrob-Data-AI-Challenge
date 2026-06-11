# Redrob-Data-AI-Challenge
This repository contains a high-performance, offline AI ranking pipeline built for the Redrob Intelligent Candidate Discovery Challenge. The objective is to evaluate a massive pool of 100,000 synthetic candidates against a highly specific "Senior AI Engineer" job description and output the top 100 best-fit matches.
# Redrob Intelligent Candidate Discovery & Ranking

## Description
This repository contains a high-performance, offline AI ranking pipeline built for the Redrob Intelligent Candidate Discovery Challenge. The objective is to evaluate a massive pool of 100,000 synthetic candidates against a highly specific "Senior AI Engineer" job description and output the top 100 best-fit matches.

Instead of relying on basic keyword matching—which falls for dataset traps—this solution bridges the gap between what a job description *says* and what it actually *means* using a dual-scoring architecture:

* **Semantic Text Matching:** Uses a TF-IDF Vectorizer (with bi-grams) to evaluate the depth of a candidate's technical experience, summary, and skills against the target job description.
* **Behavioral Intelligence:** Applies mathematical multipliers based on dynamic `redrob_signals` (e.g., heavily penalizing candidates who ghost recruiters or haven't logged in for months, while boosting active open-source contributors).
* **Honeypot Defense:** Includes explicit logic to detect and instantly disqualify non-technical candidates (like Marketing or HR roles) attempting to stuff AI keywords into their profiles.

Designed to be highly efficient, this pipeline reads directly from the dataset and executes entirely locally on a CPU, easily beating the strict 5-minute compute constraint.

## Interactive Sandbox
https://colab.research.google.com/drive/136X1XjdF-NbnLlj8wjVPwelGu2dEi5qi?usp=sharing

## How to Run Locally Offline
Ensure you have the `candidates.jsonl.gz` dataset in the root directory.
1. Install dependencies: `pip install -r requirements.txt`
2. Run the pipeline: `python ranker.py`
3. The script will generate the final CSV containing the top 100 candidate IDs and a 1-2 sentence justification for manual review.
