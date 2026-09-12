import json
import csv
import sys
from pathlib import Path

# Add intents_data to path
base_dir = Path(__file__).resolve().parent
sys.path.append(str(base_dir / "intents_data"))

from gen_battery import create_golden_dataset
from gen_software import get_software_cases
from gen_account import get_account_cases
from gen_connectivity import get_connectivity_cases
from gen_billing import get_billing_cases
from gen_repair import get_repair_cases
from gen_features import get_features_cases

def main():
    cases = []
    cases.extend(create_golden_dataset())
    cases.extend(get_software_cases())
    cases.extend(get_account_cases())
    cases.extend(get_connectivity_cases())
    cases.extend(get_billing_cases())
    cases.extend(get_repair_cases())
    cases.extend(get_features_cases())

    print(f"Total Golden Cases Assembled: {len(cases)}")
    assert len(cases) == 200, f"Expected 200 cases, got {len(cases)}"

    # Check distribution
    intents = {}
    escalations = {True: 0, False: 0}
    difficulties = {}
    for c in cases:
        intents[c["true_intent"]] = intents.get(c["true_intent"], 0) + 1
        escalations[c["true_escalate"]] += 1
        difficulties[c["difficulty"]] = difficulties.get(c["difficulty"], 0) + 1

    print("Intent Breakdown:", json.dumps(intents, indent=2))
    print("Escalation Breakdown:", json.dumps(escalations, indent=2))
    print("Difficulty Breakdown:", json.dumps(difficulties, indent=2))

    data_dir = base_dir.parents[1] / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    json_path = data_dir / "golden_eval_set.json"
    csv_path = data_dir / "golden_eval_set.csv"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(cases, f, indent=2, ensure_ascii=False)
    print(f"Successfully saved {json_path}")

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(cases[0].keys()))
        writer.writeheader()
        writer.writerows(cases)
    print(f"Successfully saved {csv_path}")

if __name__ == "__main__":
    main()
