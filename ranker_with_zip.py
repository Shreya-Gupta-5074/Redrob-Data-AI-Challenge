import json
import re
import time
import gzip
import datetime
import pandas as pd
from typing import List, Dict, Set
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ==========================================
# 1. Configuration
# ==========================================
CANDIDATES_FILE = "candidates.jsonl.gz"  # Now reads compressed directly
JOB_DESCRIPTION_FILE = "job_description.txt"
OUTPUT_CSV = "team_123.csv"  # TODO: Replace with your actual team ID
REFERENCE_DATE = datetime.datetime.now() # Used to calculate "last active" days

# ==========================================
# 2. Honeypot & Behavioral Logic
# ==========================================
def is_honeypot(title: str) -> bool:
    """Detects non-technical roles attempting keyword stuffing."""
    title_lower = str(title).lower()
    honeypot_terms = [
        'marketing', 'sales', 'hr ', 'human resources', 'recruiter', 
        'talent acquisition', 'account executive', 'financial advisor', 'seo'
    ]
    for term in honeypot_terms:
        if term in title_lower or title_lower.startswith('hr'):
            return True
    return False

def calculate_behavioral_multiplier(signals: dict) -> float:
    """
    Calculates a score multiplier based on Redrob behavioral signals.
    Baseline is 1.0. High engagement boosts it; low engagement penalizes it.
    """
    if not signals:
        return 1.0
        
    multiplier = 1.0

    # 1. Recruiter Response Rate (0.0 to 1.0)
    rr_rate = signals.get('recruiter_response_rate', 0.5)
    if rr_rate < 0.05:
        multiplier *= 0.10  # 90% penalty for ghosters
    else:
        multiplier *= (0.5 + rr_rate)  # Up to 1.5x boost for highly responsive

    # 2. Last Active Date (Penalize > 180 days)
    last_active_str = signals.get('last_active_date')
    if last_active_str:
        try:
            last_active = datetime.datetime.strptime(last_active_str, "%Y-%m-%d")
            days_inactive = abs((REFERENCE_DATE - last_active).days)
            if days_inactive > 180:
                multiplier *= 0.3  # 70% penalty for dormant accounts
        except ValueError:
            pass

    # 3. Open to Work Flag
    if not signals.get('open_to_work_flag', True):
        multiplier *= 0.8  # 20% penalty if not actively open

    # 4. GitHub Activity Score (Bonus for engineers)
    github_score = signals.get('github_activity_score', 0)
    if github_score > 5.0:
        multiplier *= 1.1  # 10% bonus for strong open-source/coding signals

    return multiplier

# ==========================================
# 3. Reasoning Generator
# ==========================================
def generate_reasoning(cand_row: pd.Series, jd_keywords: set) -> str:
    """Generates context-aware reasoning for Stage 4 manual review."""
    score = cand_row['final_score']
    yoe = cand_row['yoe']
    cand_skills = set([s.lower() for s in cand_row['raw_skills']])
    matched_skills = list(jd_keywords.intersection(cand_skills))
    
    if cand_row['is_honeypot']:
        return "Rejected: Flagged as a non-technical honeypot profile."
        
    if score > 0.05:
        reason = f"Strong fit with {yoe} YoE and excellent behavioral signals. "
        if matched_skills:
            top_skills = ", ".join([s.title() for s in matched_skills[:3]])
            reason += f"Demonstrates core technical competencies in {top_skills}."
        return reason.strip()
    return f"Selected based on baseline semantic alignment and active profile metrics. YoE: {yoe}."

# ==========================================
# 4. Main Processing Pipeline
# ==========================================
def main():
    start_time = time.time()
    print("🚀 Starting Production Candidate Screening...")

    # A. Read Job Description
    try:
        with open(JOB_DESCRIPTION_FILE, 'r', encoding='utf-8') as f:
            jd_text = f.read()
    except FileNotFoundError:
        print(f"⚠️ {JOB_DESCRIPTION_FILE} not found.")
        return

    jd_keywords = set(re.findall(r'\b[a-zA-Z]{3,}\b', jd_text.lower()))

    # B. Load Candidates from GZIP
    candidates = []
    print(f"Unpacking and parsing {CANDIDATES_FILE} on the fly...")
    
    try:
        # 🔑 FIX: Reading .gz files directly
        with gzip.open(CANDIDATES_FILE, 'rt', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                cand = json.loads(line)
                
                cand_id = cand.get('candidate_id', '')
                profile = cand.get('profile', {})
                current_title = str(profile.get('current_title', ''))
                
                # Check Honeypot
                honeypot_flag = is_honeypot(current_title)
                
                # Extract Signals
                signals = cand.get('redrob_signals', {})
                behavior_mult = calculate_behavioral_multiplier(signals)
                
                # Text Extraction
                headline = str(profile.get('headline') or "")
                summary = str(profile.get('summary') or "")
                yoe = profile.get('years_of_experience', 0.0)
                
                history = cand.get('career_history', [])
                job_texts = [str(job.get('title') or "") + " " + str(job.get('description') or "") for job in history]
                
                skills = cand.get('skills', [])
                skill_names = [str(skill.get('name') or "") for skill in skills]
                
                # Combine Text
                combined_text_parts = [headline, summary] + job_texts + skill_names + skill_names
                clean_text = " ".join([t for t in combined_text_parts if t.strip()])
                
                candidates.append({
                    "candidate_id": cand_id,
                    "clean_text": clean_text,
                    "yoe": yoe,
                    "raw_skills": skill_names,
                    "is_honeypot": honeypot_flag,
                    "behavior_mult": behavior_mult
                })
                
    except FileNotFoundError:
        print(f"⚠️ {CANDIDATES_FILE} not found. Make sure it's in the same directory.")
        return

    df = pd.DataFrame(candidates)
    if df.empty:
        print("No candidates loaded.")
        return

    print(f"✅ Loaded {len(df)} candidates. Processing NLP vectors...")

    # C. TF-IDF Semantic Scoring
    vectorizer = TfidfVectorizer(stop_words='english', max_features=20000, ngram_range=(1, 2))
    all_texts = [jd_text] + df['clean_text'].tolist()
    tfidf_matrix = vectorizer.fit_transform(all_texts)
    
    jd_vector = tfidf_matrix[0]
    candidate_vectors = tfidf_matrix[1:]
    base_scores = cosine_similarity(jd_vector, candidate_vectors)[0]
    
    df['base_score'] = base_scores.astype(float)

    # D. Apply Behavioral Multipliers & Penalties
    print("Applying Redrob behavioral signals and filtering honeypots...")
    df['final_score'] = df['base_score'] * df['behavior_mult']
    
    # 🔑 FIX: Zero out honeypots completely
    df.loc[df['is_honeypot'] == True, 'final_score'] = 0.0

    # E. Strict Ranking
    print("Sorting and generating strict Top 100 ranks...")
    df = df.sort_values(by=['final_score', 'candidate_id'], ascending=[False, True]).reset_index(drop=True)
    
    top_100 = df.head(100).copy()
    top_100['rank'] = range(1, len(top_100) + 1)

    # F. Justifications
    print("Drafting reasoning sentences...")
    top_100['reasoning'] = top_100.apply(lambda row: generate_reasoning(row, jd_keywords), axis=1)

    # G. Export strictly formatted CSV
    print(f"Writing final submission to {OUTPUT_CSV}...")
    
    # 🔑 FIX: Guarantee EXACT requested column names. Rename final_score back to score.
    top_100 = top_100.rename(columns={'final_score': 'score'})
    output_cols = ['candidate_id', 'rank', 'score', 'reasoning']
    final_df = top_100[output_cols]
    
    final_df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8')

    elapsed = time.time() - start_time
    print(f"✅ Pipeline perfectly completed in {elapsed:.2f} seconds!")
    print("\n--- TOP 3 SELECTED CANDIDATES ---")
    print(final_df[['candidate_id', 'score']].head(3).to_string(index=False))

if __name__ == "__main__":
    main()