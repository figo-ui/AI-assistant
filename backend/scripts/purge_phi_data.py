import argparse
import json
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Delete expired PHI-bearing case and audit records.")
    parser.add_argument("--case-retention-days", type=int, default=30)
    parser.add_argument("--audit-retention-days", type=int, default=365)
    args = parser.parse_args()

    backend_dir = Path(__file__).resolve().parents[1]
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "healthcare_ai.settings")

    import sys

    sys.path.insert(0, str(backend_dir))

    import django

    django.setup()

    from guidance.services.retention import purge_expired_phi_records

    summary = purge_expired_phi_records(
        case_retention_days=args.case_retention_days,
        audit_retention_days=args.audit_retention_days,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
