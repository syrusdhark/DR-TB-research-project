"""
Build the Stage 2 (DR-TB risk model) training table.

Joins data/clinical_data.csv and data/genomic_mutations.csv on patient_id
(4,200 unique patients, one row each -- no image involved at all).

The original merged_dataset.csv's label_drtb column contradicts the very
fields it should be derived from: patients with both MDR-TB and XDR-TB
confirmed are labeled DR-TB-negative 100% of the time (0/10), confirmed
XDR-TB alone is labeled negative 98.6% of the time, and 5+ resistance
mutations is labeled negative 100% of the time. That is inverted from basic
TB domain knowledge, so it isn't used here.

Instead, label_drtb is derived deterministically and consistently with the
input features:
    label_drtb = 1  if  mdr_tb OR xdr_tb OR rifampin_resistance
                        OR isoniazid_resistance OR mutation_count >= 1
                 0  otherwise
"""

from pathlib import Path
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
CLINICAL_CSV = REPO_ROOT / "data" / "clinical_data.csv"
GENOMIC_CSV = REPO_ROOT / "data" / "genomic_mutations.csv"
OUTPUT_CSV = REPO_ROOT / "data" / "drtb_risk_dataset.csv"

REGIONS = ["Africa", "Americas", "Asia", "Europe"]


def derive_label(row) -> int:
    return int(
        row["mdr_tb"] == 1
        or row["xdr_tb"] == 1
        or row["rifampin_resistance"] == 1
        or row["isoniazid_resistance"] == 1
        or row["mutation_count"] >= 1
    )


def main():
    clinical = pd.read_csv(CLINICAL_CSV)
    genomic = pd.read_csv(GENOMIC_CSV)

    merged = clinical.merge(genomic, on="patient_id", how="inner", validate="one_to_one")
    assert len(merged) == len(clinical) == len(genomic), "Expected a clean 1:1 join"

    merged["gender_encoded"] = (merged["gender"] == "M").astype(int)
    for region in REGIONS:
        merged[f"region_{region}"] = (merged["region"] == region).astype(int)

    merged["label_drtb"] = merged.apply(derive_label, axis=1)

    merged.to_csv(OUTPUT_CSV, index=False)

    print(f"Wrote {len(merged)} patient rows to {OUTPUT_CSV}")
    print(f"  label_drtb positive rate: {merged['label_drtb'].mean():.3f}")
    print(f"  positives: {merged['label_drtb'].sum()}  negatives: {(1 - merged['label_drtb']).sum()}")


if __name__ == "__main__":
    main()
