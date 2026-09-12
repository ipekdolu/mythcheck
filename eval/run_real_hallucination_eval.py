"""Phase 5: run the small, manually-checked real-hallucination set through the
pipeline. This is a sanity check against the synthetic corrupted-variant set -
synthetic corruption catches "planted" errors cleanly, but real hallucinations
aren't always as clean, so this confirms the pipeline generalizes.

Gold labels here were assigned by checking each generated sentence directly
against the pulled Wikipedia source text in data/raw/ (see check_note on each
item) - not an independent human annotator, which is disclosed in the README.

Run with: python -m eval.run_real_hallucination_eval
"""
import json
import pathlib

from verification.pipeline import verify_text

SET_PATH = pathlib.Path("eval/real_hallucination_set.json")
RESULTS_PATH = pathlib.Path("eval/real_hallucination_results.json")


def _sentence_level_verdict(verdicts: list[dict]) -> str:
    """A single input sentence often decomposes into several atomic claims
    (e.g. "X and Y are Z's sons" -> one claim per name). Roll those back up:
    if any sub-claim is CONTRADICTED, the sentence contains a hallucination;
    otherwise if any is SUPPORTED, treat the sentence as supported; else
    UNVERIFIABLE. This matches how a reader would judge the sentence as a
    whole, and avoids the wrong signal from checking only the first
    extracted claim (verified against this exact set - see eval/README or
    commit history for the diagnostic that caught this)."""
    kinds = {v["verdict"] for v in verdicts}
    if "CONTRADICTED" in kinds:
        return "CONTRADICTED"
    if kinds == {"SUPPORTED"}:
        return "SUPPORTED"
    return "UNVERIFIABLE"  # any unconfirmed sub-claim keeps the whole sentence unconfirmed


def run() -> list[dict]:
    items = json.loads(SET_PATH.read_text(encoding="utf-8"))
    results = []
    for i, item in enumerate(items, 1):
        pipeline_result = verify_text(item["text"])
        verdicts = pipeline_result["verdicts"]
        predicted = _sentence_level_verdict(verdicts) if verdicts else "UNVERIFIABLE"
        results.append({**item, "predicted_verdict": predicted, "sub_claims": verdicts})
        print(f"[{i}/{len(items)}] gold={item['gold_verdict']:13} pred={predicted:13} {item['text'][:70]}")
    RESULTS_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    return results


def score(results: list[dict]) -> None:
    tp = sum(1 for r in results if r["gold_verdict"] == "CONTRADICTED" and r["predicted_verdict"] == "CONTRADICTED")
    fp = sum(1 for r in results if r["gold_verdict"] == "SUPPORTED" and r["predicted_verdict"] == "CONTRADICTED")
    fn = sum(1 for r in results if r["gold_verdict"] == "CONTRADICTED" and r["predicted_verdict"] != "CONTRADICTED")
    genuine = [r for r in results if r["gold_verdict"] == "SUPPORTED"]
    fp_rate = sum(1 for r in genuine if r["predicted_verdict"] == "CONTRADICTED") / len(genuine) if genuine else float("nan")

    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    print(f"\nn={len(results)} precision={precision:.2f} recall={recall:.2f} false-positive-rate={fp_rate:.2f}")


if __name__ == "__main__":
    results = run()
    score(results)
