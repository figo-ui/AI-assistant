import uuid

from django.db import migrations, models


def populate_status_tokens(apps, schema_editor):
    CaseSubmission = apps.get_model("guidance", "CaseSubmission")
    for case in CaseSubmission.objects.filter(status_token__isnull=True).iterator():
        case.status_token = uuid.uuid4()
        case.save(update_fields=["status_token"])


class Migration(migrations.Migration):
    dependencies = [
        ("guidance", "0002_healthcarefacility_alter_casesubmission_options_and_more"),
    ]

    operations = [
        migrations.AddField(
            model_name="casesubmission",
            name="async_job_id",
            field=models.CharField(blank=True, default="", max_length=64),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name="casesubmission",
            name="status_token",
            field=models.UUIDField(blank=True, db_index=True, editable=False, null=True),
        ),
        migrations.RunPython(populate_status_tokens, migrations.RunPython.noop),
        migrations.AlterField(
            model_name="casesubmission",
            name="status_token",
            field=models.UUIDField(db_index=True, default=uuid.uuid4, editable=False, unique=True),
        ),
    ]
