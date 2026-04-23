from __future__ import annotations

from datetime import timedelta
from typing import Dict

from django.db import transaction
from django.utils import timezone
from django.utils.dateparse import parse_datetime

from ..models import AuditLog, CaseSubmission


def _coerce_now(now=None):
    if now is None:
        return timezone.now()
    if hasattr(now, "tzinfo"):
        return now
    parsed = parse_datetime(str(now))
    if parsed is None:
        raise ValueError("now must be a datetime or ISO-8601 string")
    if timezone.is_naive(parsed):
        return timezone.make_aware(parsed, timezone.utc)
    return parsed


@transaction.atomic
def purge_expired_phi_records(
    *,
    case_retention_days: int,
    audit_retention_days: int = 365,
    now=None,
) -> Dict[str, int]:
    current_time = _coerce_now(now)
    case_cutoff = current_time - timedelta(days=max(1, int(case_retention_days)))
    audit_cutoff = current_time - timedelta(days=max(1, int(audit_retention_days)))

    expired_case_ids = list(
        CaseSubmission.objects.filter(created_at__lt=case_cutoff).values_list("id", flat=True)
    )
    cases_deleted, _ = CaseSubmission.objects.filter(id__in=expired_case_ids).delete()
    audit_deleted, _ = AuditLog.objects.filter(created_at__lt=audit_cutoff).delete()

    return {
        "cases_deleted": int(len(expired_case_ids)),
        "case_rows_deleted": int(cases_deleted),
        "audit_rows_deleted": int(audit_deleted),
    }
