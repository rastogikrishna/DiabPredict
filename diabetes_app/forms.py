from django import forms
from .models import PredictionHistory

class PredictionForm(forms.ModelForm):
    class Meta:
        model = PredictionHistory
        fields = [
            'pregnancies', 'glucose', 'blood_pressure', 
            'skin_thickness', 'insulin', 'bmi', 'diabetes_pedigree_function', 'age'
        ]
        widgets = {
            'pregnancies': forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'e.g. 0', 'min': 0}),
            'glucose': forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'e.g. 120 (mg/dL)', 'min': 0, 'step': 'any'}),
            'blood_pressure': forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'e.g. 80 (mmHg)', 'min': 0, 'step': 'any'}),
            'skin_thickness': forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'e.g. 20 (mm)', 'min': 0, 'step': 'any'}),
            'insulin': forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'e.g. 85 (µU/mL)', 'min': 0, 'step': 'any'}),
            'bmi': forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'e.g. 25.4', 'min': 0, 'step': 'any'}),
            'diabetes_pedigree_function': forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'e.g. 0.471', 'min': 0, 'step': 'any'}),
            'age': forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'e.g. 33', 'min': 0}),
        }

    def clean_pregnancies(self):
        val = self.cleaned_data.get('pregnancies')
        if val is not None and val < 0:
            raise forms.ValidationError("Pregnancies cannot be negative.")
        return val

    def clean_glucose(self):
        val = self.cleaned_data.get('glucose')
        if val is not None and val < 0:
            raise forms.ValidationError("Glucose cannot be negative.")
        return val

    def clean_blood_pressure(self):
        val = self.cleaned_data.get('blood_pressure')
        if val is not None and val < 0:
            raise forms.ValidationError("Blood Pressure cannot be negative.")
        return val

    def clean_skin_thickness(self):
        val = self.cleaned_data.get('skin_thickness')
        if val is not None and val < 0:
            raise forms.ValidationError("Skin Thickness cannot be negative.")
        return val

    def clean_insulin(self):
        val = self.cleaned_data.get('insulin')
        if val is not None and val < 0:
            raise forms.ValidationError("Insulin cannot be negative.")
        return val

    def clean_bmi(self):
        val = self.cleaned_data.get('bmi')
        if val is not None and val < 0:
            raise forms.ValidationError("BMI cannot be negative.")
        return val

    def clean_diabetes_pedigree_function(self):
        val = self.cleaned_data.get('diabetes_pedigree_function')
        if val is not None and val < 0:
            raise forms.ValidationError("Diabetes Pedigree Function cannot be negative.")
        return val

    def clean_age(self):
        val = self.cleaned_data.get('age')
        if val is not None and val < 0:
            raise forms.ValidationError("Age cannot be negative.")
        return val
