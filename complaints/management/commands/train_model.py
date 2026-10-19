"""
Django management command to train and serialize the CampusCare category model.
Usage:
    python manage.py train_model
    python manage.py train_model --data path/to/complaints.csv --output path/to/category_model.joblib
"""
import os
from django.core.management.base import BaseCommand
from complaints.ml import train_category_model, MODEL_PATH, DATA_PATH


class Command(BaseCommand):
    help = "Train and serialize the CampusCare category classification model."

    def add_arguments(self, parser):
        parser.add_argument(
            "--data",
            default=DATA_PATH,
            help="Path to training complaints.csv dataset (default: data/complaints.csv)"
        )
        parser.add_argument(
            "--output",
            default=MODEL_PATH,
            help="Path to save the joblib model artifact (default: data/category_model.joblib)"
        )

    def handle(self, *args, **options):
        data_path = options["data"]
        output_path = options["output"]

        if not os.path.exists(data_path):
            self.stderr.write(self.style.ERROR(f"Data file not found at {data_path}"))
            return

        self.stdout.write(f"Training category model from {data_path}...")
        train_category_model(data_path=data_path, save_path=output_path)
        self.stdout.write(
            self.style.SUCCESS(f"Successfully trained and saved model to {output_path} ({os.path.getsize(output_path)} bytes)")
        )
