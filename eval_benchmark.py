import os
import json
import django
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
django.setup()

from complaints.ml import (
    predict_category,
    predict_category_with_confidence,
    predict_priority
)

with open("data/eval_benchmark.json", "r", encoding="utf-8") as f:
    benchmark_data = json.load(f)

true_categories = [item["category"] for item in benchmark_data]
true_priorities = [item["priority"] for item in benchmark_data]

pred_categories = []
confidences = []
pred_priorities = []

for item in benchmark_data:
    cat, conf = predict_category_with_confidence(item["complaint"])
    pri = predict_priority(item["complaint"])
    pred_categories.append(cat)
    confidences.append(conf)
    pred_priorities.append(pri)

cat_acc = sum(t == p for t, p in zip(true_categories, pred_categories)) / len(benchmark_data)
pri_acc = sum(t == p for t, p in zip(true_priorities, pred_priorities)) / len(benchmark_data)
avg_conf = sum(confidences) / len(confidences)

print("=" * 60)
print("CAMPUSCARE EVALUATION BENCHMARK RESULTS (50 samples)")
print("=" * 60)
print(f"Category Prediction Accuracy: {cat_acc * 100:.2f}% ({sum(t == p for t, p in zip(true_categories, pred_categories))}/{len(benchmark_data)})")
print(f"Mean Prediction Confidence:    {avg_conf * 100:.2f}%")
print(f"Priority Prediction Accuracy: {pri_acc * 100:.2f}% ({sum(t == p for t, p in zip(true_priorities, pred_priorities))}/{len(benchmark_data)})")
print("=" * 60)
print("\n--- Category Classification Report ---")
print(classification_report(true_categories, pred_categories, digits=3))

print("\n--- Priority Classification Report ---")
print(classification_report(true_priorities, pred_priorities, digits=3))
