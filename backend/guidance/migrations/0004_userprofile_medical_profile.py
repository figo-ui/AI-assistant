from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("guidance", "0003_casesubmission_async_job_id_casesubmission_status_token"),
    ]

    operations = [
        migrations.AddField(
            model_name="userprofile",
            name="medical_profile",
            field=models.JSONField(blank=True, default=dict),
        ),
    ]
