from __future__ import annotations

from typing import Optional

import django_rq
from django.conf import settings
from django.db import close_old_connections
from rq.job import Job

from ..models import CaseSubmission
from .pipeline import run_case_analysis


def execute_case_analysis(case_id: int) -> None:
    close_old_connections()
    case = CaseSubmission.objects.get(id=case_id)
    try:
        run_case_analysis(case)
    except Exception:
        case.status = "failed"
        case.save(update_fields=["status"])
        raise
    finally:
        close_old_connections()


def submit_async_case_analysis(case_id: int):
    queue = django_rq.get_queue(getattr(settings, "RQ_ANALYSIS_QUEUE", "analysis"))
    job = queue.enqueue(
        execute_case_analysis,
        case_id,
        job_timeout=int(getattr(settings, "RQ_JOB_TIMEOUT", 600)),
        failure_ttl=24 * 3600,
        result_ttl=24 * 3600,
    )
    CaseSubmission.objects.filter(id=case_id).update(async_job_id=job.id, status="queued")
    return job


def _fetch_job(job_id: str) -> Optional[Job]:
    if not job_id:
        return None
    try:
        return Job.fetch(job_id, connection=django_rq.get_connection(getattr(settings, "RQ_ANALYSIS_QUEUE", "analysis")))
    except Exception:
        return None


def async_case_status(case_id: int) -> str:
    case = CaseSubmission.objects.filter(id=case_id).only("status", "async_job_id").first()
    if case is None:
        return "unknown"
    if not case.async_job_id:
        return case.status or "unknown"

    job = _fetch_job(case.async_job_id)
    if job is None:
        return case.status or "unknown"
    status_value = job.get_status(refresh=False)
    if status_value == "failed":
        return "failed"
    if status_value == "finished":
        return "completed"
    if status_value == "started":
        return "running"
    if status_value == "queued":
        return "queued"
    return case.status or "unknown"
