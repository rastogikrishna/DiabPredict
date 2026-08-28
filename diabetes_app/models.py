from django.db import models

class PredictionHistory(models.Model):
    timestamp = models.DateTimeField(auto_now_add=True, verbose_name="Assessment Date")
    pregnancies = models.IntegerField(verbose_name="Pregnancies")
    glucose = models.FloatField(verbose_name="Glucose")
    blood_pressure = models.FloatField(verbose_name="Blood Pressure")
    skin_thickness = models.FloatField(verbose_name="Skin Thickness")
    insulin = models.FloatField(verbose_name="Insulin")
    bmi = models.FloatField(verbose_name="BMI")
    diabetes_pedigree_function = models.FloatField(verbose_name="Diabetes Pedigree Function")
    age = models.IntegerField(verbose_name="Age")
    
    # ML Results
    predicted_class = models.IntegerField(verbose_name="Predicted Class")
    probability = models.FloatField(verbose_name="Risk Probability")
    risk_level = models.CharField(max_length=15, verbose_name="Risk Level")

    class Meta:
        ordering = ['-timestamp']
        verbose_name_plural = "Prediction Histories"

    def __str__(self):
        outcome = "Diabetic" if self.predicted_class == 1 else "Healthy"
        return f"Assessment on {self.timestamp.strftime('%Y-%m-%d %H:%M')} - Outcome: {outcome} ({self.risk_level} Risk)"
