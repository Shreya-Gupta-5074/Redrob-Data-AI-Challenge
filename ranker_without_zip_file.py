import json
import re
import time
import pandas as pd
from typing import List, Dict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ==========================================
# 1. Configuration
# ==========================================
CANDIDATES_FILE = "candidates.jsonl"
JOB_DESCRIPTION_FILE = "job_description.txt" # Make sure you have this saved!
OUTPUT_CSV = "team_123.csv" # TODO: Replace with your actual team ID

# ==========================================
# 2. Reasoning Generator
# ==========================================
def generate_reasoning(cand_row: pd.Series, jd_keywords: set) -> str:
    """
    Generates a realistic 1-2 sentence justification for Stage 4 manual review.
    Uses actual skills and experience from the candidate's parsed data.
    """
    score = cand_row['score']
    yoe = cand_row['yoe']
    cand_skills = set([s.lower() for s in cand_row['raw_skills']])
    
    # Find matching skills between Job Description and Candidate's skills
    matched_skills = list(jd_keywords.intersection(cand_skills))
    
    if score > 0.05:
        reason = f"Strong candidate with {yoe} years of experience and high semantic match to the job description (score: {score:.3f}). "
        if matched_skills:
            top_skills = ", ".join([s.title() for s in matched_skills[:3]])
            reason += f"Demonstrates core competencies in key required areas including {top_skills}."
        return reason.strip()
    elif score > 0.0:
        return f"Selected based on partial semantic alignment (score: {score:.3f}) and {yoe} years of professional experience. Meets baseline text structural requirements."
    else:
        return "Included as rank filler to meet the top 100 requirement. Minimal semantic overlap with target job description."

# ==========================================
# 3. Main Processing Pipeline
# ==========================================
def main():
    start_time = time.time()
    print("🚀 Starting offline intelligent candidate screening...")

    # A. Read Job Description
    try:
        with open(JOB_DESCRIPTION_FILE, 'r', encoding='utf-8') as f:
            jd_text = f.read()
    except FileNotFoundError:
        print(f"⚠️ {JOB_DESCRIPTION_FILE} not found. Please create it with the job text.")
        return

    # Extract JD keywords for reasoning (words 3+ chars)
    jd_words = re.findall(r'\b[a-zA-Z]{3,}\b', jd_text.lower())
    jd_keywords = set(jd_words)

    # B. Load Candidates JSONL (TARGETED EXTRACTION)
    candidates = []
    print("Parsing candidates.jsonl intelligently...")
    
    try:
        with open(CANDIDATES_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                cand = json.loads(line)
                
                cand_id = cand.get('candidate_id', '')
                
                # 1. Profile Data
                profile = cand.get('profile', {})
                headline = str(profile.get('headline') or "")
                summary = str(profile.get('summary') or "")
                yoe = profile.get('years_of_experience', 0.0)
                
                # 2. Career History Data
                history = cand.get('career_history', [])
                job_texts = []
                for job in history:
                    job_texts.append(str(job.get('title') or ""))
                    job_texts.append(str(job.get('description') or ""))
                
                # 3. Skills Data
                skills = cand.get('skills', [])
                skill_names = [str(skill.get('name') or "") for skill in skills]
                
                # COMBINE TARGETED TEXT
                # We boost the importance of skills by repeating them in the text twice
                combined_text_parts = [headline, summary] + job_texts + skill_names + skill_names
                clean_text = " ".join([t for t in combined_text_parts if t.strip()])
                
                candidates.append({
                    "candidate_id": cand_id,
                    "clean_text": clean_text,
                    "yoe": yoe,
                    "raw_skills": skill_names
                })
                
    except FileNotFoundError:
        print(f"⚠️ {CANDIDATES_FILE} not found.")
        return

    df = pd.DataFrame(candidates)
    if df.empty:
        print("No candidates loaded.")
        return

    print(f"✅ Successfully loaded {len(df)} candidates.")

    # C. Fast CPU Scoring (TF-IDF & Cosine Similarity)
    print("Scoring candidates via TF-IDF Vectorization...")
    # Using bi-grams (1, 2) helps capture phrases like "machine learning" or "vector search"
    vectorizer = TfidfVectorizer(stop_words='english', max_features=20000, ngram_range=(1, 2))
    
    # Build vocabulary on JD + Candidate text
    all_texts = [jd_text] + df['clean_text'].tolist()
    tfidf_matrix = vectorizer.fit_transform(all_texts)
    
    # Calculate Similarity
    jd_vector = tfidf_matrix[0]
    candidate_vectors = tfidf_matrix[1:]
    scores = cosine_similarity(jd_vector, candidate_vectors)[0]
    
    df['score'] = scores.astype(float)

    # D. Rank and Break Ties Deterministically
    print("Sorting and assigning strict ranks...")
    # Sort primarily by score (Descending). If tied, sort by candidate_id (Ascending)
    df = df.sort_values(by=['score', 'candidate_id'], ascending=[False, True]).reset_index(drop=True)
    
    # ENFORCE EXACTLY 100 ROWS
    top_100 = df.head(100).copy()
    top_100['rank'] = range(1, len(top_100) + 1)

    # E. Generate Justifications
    print("Drafting reasoning sentences for Top 100...")
    top_100['reasoning'] = top_100.apply(lambda row: generate_reasoning(row, jd_keywords), axis=1)

    # F. Format and Output CSV
    print(f"Writing final submission to {OUTPUT_CSV}...")
    output_cols = ['candidate_id', 'rank', 'score', 'reasoning']
    final_df = top_100[output_cols]
    
    final_df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8')

    elapsed = time.time() - start_time
    print(f"✅ Pipeline completed in {elapsed:.2f} seconds!")
    print("\n--- TOP 3 SELECTED CANDIDATES ---")
    print(final_df[['candidate_id', 'score', 'reasoning']].head(3).to_string(index=False))

if __name__ == "__main__":
    main()