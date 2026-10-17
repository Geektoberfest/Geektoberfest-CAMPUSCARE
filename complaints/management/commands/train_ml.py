from django.core.management.base import BaseCommand

from complaints.ml import MODELS_DIR, _train_and_save


class Command(BaseCommand):
    help = (
        "Train the complaint category model from data/complaints.csv and "
        "persist it to data/ml_models/*.joblib. Run at build time so the "
        "web process only loads pre-trained artifacts (fast startup)."
    )

    def handle(self, *args, **options):
        vectorizer, model = _train_and_save()
        self.stdout.write(
            self.style.SUCCESS(
                f"Trained on {len(model.classes_)} categories "
                f"({', '.join(sorted(model.classes_))}); "
                f"vocab size={len(vectorizer.vocabulary_)}. "
                f"Saved to {MODELS_DIR}/"
            )
        )
