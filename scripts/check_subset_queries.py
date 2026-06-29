import argparse
import json
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from baseline_retrieval import attribute_to_prompt, load_ground_truth, parse_query_string
from embedding_io import load_embeddings


def parse_args():
    parser = argparse.ArgumentParser(description="Check subset queries against embeddings.")
    parser.add_argument("--query-json", default="data/celeba_subset/queries/test_embedding_celeba_evaluation.json")
    parser.add_argument("--image-embedding-dir", default="data/celeba_subset/embeddings/test")
    parser.add_argument("--text-embedding-dir", default="data/celeba_subset/embeddings")
    parser.add_argument("--output", default="data/celeba_subset/checks/query_embedding_compatibility_check.json")
    return parser.parse_args()


def main():
    args = parse_args()
    query_path = PROJECT_ROOT / args.query_json
    image_dir = PROJECT_ROOT / args.image_embedding_dir
    text_dir = PROJECT_ROOT / args.text_embedding_dir
    output_path = PROJECT_ROOT / args.output

    entries = load_ground_truth(query_path)
    _, image_ids = load_embeddings(image_dir, "image_embeddings", "cpu")
    _, text_ids = load_embeddings(text_dir, "text_embeddings", "cpu")
    image_id_set = {int(image_id) for image_id in image_ids}
    text_id_set = {str(prompt) for prompt in text_ids}

    missing_references = []
    missing_targets = []
    empty_references = []
    missing_prompts = []
    total_references = 0
    total_targets = 0

    for entry in entries:
        positive, negative = parse_query_string(entry["query"])
        for attribute in positive + negative:
            prompt = attribute_to_prompt(attribute)
            if prompt not in text_id_set:
                missing_prompts.append({"query": entry["query"], "attribute": attribute, "prompt": prompt})

        for reference, targets in entry["ground_truth"].items():
            reference = int(reference)
            total_references += 1
            if reference not in image_id_set:
                missing_references.append({"query": entry["query"], "reference": reference})
            if len(targets) == 0:
                empty_references.append({"query": entry["query"], "reference": reference})
            for target in targets:
                total_targets += 1
                target = int(target)
                if target not in image_id_set:
                    missing_targets.append({"query": entry["query"], "reference": reference, "target": target})

    status = "pass" if not (missing_references or missing_targets or empty_references or missing_prompts) else "fail"
    report = {
        "status": status,
        "query_json": args.query_json,
        "image_embedding_dir": args.image_embedding_dir,
        "text_embedding_dir": args.text_embedding_dir,
        "query_count": len(entries),
        "total_references": total_references,
        "total_targets": total_targets,
        "image_embedding_id_count": len(image_id_set),
        "text_embedding_id_count": len(text_id_set),
        "missing_reference_count": len(missing_references),
        "missing_target_count": len(missing_targets),
        "empty_reference_count": len(empty_references),
        "missing_prompt_count": len(missing_prompts),
        "examples": {
            "missing_references": missing_references[:10],
            "missing_targets": missing_targets[:10],
            "empty_references": empty_references[:10],
            "missing_prompts": missing_prompts[:10],
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"{status}: wrote {output_path}")


if __name__ == "__main__":
    main()
