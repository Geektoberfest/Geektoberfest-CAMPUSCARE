from django.db import models


class Complaint(models.Model):
    student_name = models.CharField(max_length=100)
    description = models.TextField()
    category = models.CharField(max_length=50, default="Pending")
    priority = models.CharField(max_length=20, default="Normal")
    department = models.CharField(max_length=50, default="Pending")
    status = models.CharField(max_length=20, default="Open")
    is_duplicate = models.BooleanField(default=False)
    confidence = models.FloatField(default=1.0)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.student_name} - {self.category}"