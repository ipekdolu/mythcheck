"""Phase 5: run the eval set through the full verification pipeline and score
precision/recall by claim type and corruption type.

Run with: python -m eval.run_eval
"""
import json
import pathlib

from verification.pipeline import verify_text

EVAL_SET_PATH = pathlib.Path("eval/eval_set.json")
RESULTS_PATH = pathlib.Path("eval/eval_results.json")


def _sentence_level_verdict(verdicts: list[dict]) -> str:
    """Roll multiple extracted sub-claims for one input sentence up to a
    single sentence-level verdict: CONTRADICTED if any sub-claim is, else
    SUPPORTED if any is, else UNVERIFIABLE. See eval/run_real_hallucination_eval.py
    for why checking only the first extracted claim is wrong."""
    kinds = {v["verdict"] for v in verdicts}
    if "CONTRADICTED" in kinds:
        return "CONTRADICTED"
    if kinds == {"SUPPORTED"}:
        return "SUPPORTED"
    return "UNVERIFIABLE"  # any unconfirmed sub-claim keeps the whole sentence unconfirmed


def run() -> list[dict]:
    items = json.loads(EVAL_SET_PATH.read_text(encoding="utf-8"))
    results = []
    for i, item in enumerate(items, 1):
        pipeline_result = verify_text(item["text"])
        verdicts = pipeline_result["verdicts"]
        predicted = _sentence_level_verdict(verdicts) if verdicts else "UNVERIFIABLE"
        results.append({**item, "predicted_verdict": predicted, "n_extracted_claims": len(verdicts)})
        print(f"[{i}/{len(items)}] gold={item['gold_verdict']:13} pred={predicted:13} {item['text'][:70]}")

    RESULTS_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    return results


def score(results: list[dict]) -> None:
    def precision_recall(subset: list[dict]) -> tuple[float, float, int]:
        tp = sum(1 for r in subset if r["gold_verdict"] == "CONTRADICTED" and r["predicted_verdict"] == "CONTRADICTED")
        fp = sum(1 for r in subset if r["gold_verdict"] == "SUPPORTED" and r["predicted_verdict"] == "CONTRADICTED")
        fn = sum(1 for r in subset if r["gold_verdict"] == "CONTRADICTED" and r["predicted_verdict"] != "CONTRADICTED")
        precision = tp / (tp + fp) if (tp + fp) else float("nan")
        recall = tp / (tp + fn) if (tp + fn) else float("nan")
        return precision, recall, len(subset)

    print("\n=== Overall ===")
    p, r, n = precision_recall(results)
    print(f"  n={n} precision={p:.2f} recall={r:.2f}")

    genuine = [r for r in results if r["gold_verdict"] == "SUPPORTED"]
    fp_rate = sum(1 for r in genuine if r["predicted_verdict"] == "CONTRADICTED") / len(genuine) if genuine else float("nan")
    print(f"  false-positive rate on genuine claims: {fp_rate:.2f} (n={len(genuine)})")

    print("\n=== By claim type ===")
    for claim_type in ["relational", "descriptive"]:
        subset = [r for r in results if r["claim_type"] == claim_type]
        p, r_, n = precision_recall(subset)
        print(f"  {claim_type:12} n={n:3} precision={p:.2f} recall={r_:.2f}")

    print("\n=== By corruption type ===")
    corruption_types = sorted({r["corruption_type"] for r in results} - {"none"})
    for ct in corruption_types:
        subset = [r for r in results if r["corruption_type"] == ct]
        p, r_, n = precision_recall(subset)
        print(f"  {ct:22} n={n:3} precision={p:.2f} recall={r_:.2f}")


if __name__ == "__main__":
    results = run()
    score(results)
