import os

from django.contrib.auth.models import User
from django.core.management.base import BaseCommand, CommandError


class Command(BaseCommand):
    help = "Create or update a seeded admin/superuser account."

    def add_arguments(self, parser):
        parser.add_argument("--username", default=os.getenv("SEED_ADMIN_USERNAME", "admin"))
        parser.add_argument("--email", default=os.getenv("SEED_ADMIN_EMAIL", "admin@example.com"))
        parser.add_argument("--password", default=os.getenv("SEED_ADMIN_PASSWORD"))
        parser.add_argument("--first-name", default=os.getenv("SEED_ADMIN_FIRST_NAME", "System"))
        parser.add_argument("--last-name", default=os.getenv("SEED_ADMIN_LAST_NAME", "Admin"))
        parser.add_argument(
            "--reset-password",
            action="store_true",
            help="Reset password if user already exists.",
        )

    def handle(self, *args, **options):
        username = str(options["username"]).strip()
        email = str(options["email"]).strip().lower()
        password = options.get("password")
        first_name = str(options.get("first_name", "")).strip()
        last_name = str(options.get("last_name", "")).strip()
        reset_password = bool(options.get("reset_password", False))

        if not username:
            raise CommandError("username cannot be empty")
        if not email:
            raise CommandError("email cannot be empty")

        if not password:
            raise CommandError(
                "password is required. Use --password or set SEED_ADMIN_PASSWORD env var."
            )

        user, created = User.objects.get_or_create(
            username=username,
            defaults={
                "email": email,
                "first_name": first_name,
                "last_name": last_name,
                "is_staff": True,
                "is_superuser": True,
                "is_active": True,
            },
        )

        if created:
            user.set_password(password)
            user.save()
            self.stdout.write(
                self.style.SUCCESS(
                    f"Created admin user username='{username}' email='{email}'."
                )
            )
            return

        changed_fields = []
        if user.email != email:
            user.email = email
            changed_fields.append("email")
        if user.first_name != first_name:
            user.first_name = first_name
            changed_fields.append("first_name")
        if user.last_name != last_name:
            user.last_name = last_name
            changed_fields.append("last_name")
        if not user.is_staff:
            user.is_staff = True
            changed_fields.append("is_staff")
        if not user.is_superuser:
            user.is_superuser = True
            changed_fields.append("is_superuser")
        if not user.is_active:
            user.is_active = True
            changed_fields.append("is_active")

        if reset_password:
            user.set_password(password)
            changed_fields.append("password")

        if changed_fields:
            user.save()
            self.stdout.write(
                self.style.SUCCESS(
                    f"Updated admin user '{username}'. Fields: {', '.join(changed_fields)}"
                )
            )
        else:
            self.stdout.write(self.style.WARNING(f"Admin user '{username}' already up to date."))
