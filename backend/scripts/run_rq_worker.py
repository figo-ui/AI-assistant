import os
import sys
from pathlib import Path


def main() -> None:
    backend_dir = Path(__file__).resolve().parents[1]
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "healthcare_ai.settings")
    sys.path.insert(0, str(backend_dir))

    import django

    django.setup()

    import django_rq
    from django.conf import settings

    worker = django_rq.get_worker(getattr(settings, "RQ_ANALYSIS_QUEUE", "analysis"))
    worker.work(with_scheduler=False)


if __name__ == "__main__":
    main()
