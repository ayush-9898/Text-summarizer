# ============================================================
# evaluate.py
# PURPOSE: Measure the quality of generated summaries.
#
# ROUGE = Recall-Oriented Understudy for Gisting Evaluation
# It compares your generated summary to a reference (gold) summary
# by counting word overlaps.
#
# ROUGE-1: Overlap of single words
# ROUGE-2: Overlap of word pairs (bigrams)
# ROUGE-L: Longest common subsequence
# ============================================================

from rouge_score import rouge_scorer


def evaluate_summary(
    generated_summary: str,
    reference_summary: str,
) -> dict:
    """
    Calculate ROUGE scores between generated and reference summaries.

    Args:
        generated_summary : The summary your model produced
        reference_summary : The "gold standard" / human-written summary

    Returns:
        Dict with ROUGE-1, ROUGE-2, ROUGE-L scores
        Each score has: precision, recall, f1 (use f1 as main metric)

    What the scores mean:
        0.0 - 0.2  : Poor
        0.2 - 0.4  : Fair
        0.4 - 0.6  : Good
        0.6 - 1.0  : Excellent
    """
    scorer = rouge_scorer.RougeScorer(
        ['rouge1', 'rouge2', 'rougeL'],
        use_stemmer=True,
    )

    scores = scorer.score(reference_summary, generated_summary)

    return {
        "rouge1": {
            "precision": round(scores['rouge1'].precision, 4),
            "recall":    round(scores['rouge1'].recall,    4),
            "f1":        round(scores['rouge1'].fmeasure,  4),
        },
        "rouge2": {
            "precision": round(scores['rouge2'].precision, 4),
            "recall":    round(scores['rouge2'].recall,    4),
            "f1":        round(scores['rouge2'].fmeasure,  4),
        },
        "rougeL": {
            "precision": round(scores['rougeL'].precision, 4),
            "recall":    round(scores['rougeL'].recall,    4),
            "f1":        round(scores['rougeL'].fmeasure,  4),
        },
    }


def print_evaluation(scores: dict) -> None:
    """Pretty-print the evaluation results."""
    print("\n📊 ROUGE Evaluation Results:")
    print("="*40)
    for metric, values in scores.items():
        print(f"  {metric.upper()}")
        print(f"    Precision : {values['precision']:.4f}")
        print(f"    Recall    : {values['recall']:.4f}")
        print(f"    F1 Score  : {values['f1']:.4f} ⬅ main metric")
        print()


# ── Quick test ──────────────────────────────────────────────
if __name__ == "__main__":
    generated = (
        "AI is transforming healthcare and transportation. "
        "Machine learning models detect cancer with high accuracy. "
        "Challenges include bias and job displacement."
    )

    reference = (
        "Artificial intelligence has transformed many industries. "
        "In healthcare, AI detects diseases from images. "
        "Concerns about algorithmic bias and employment remain."
    )

    scores = evaluate_summary(generated, reference)
    print_evaluation(scores)
