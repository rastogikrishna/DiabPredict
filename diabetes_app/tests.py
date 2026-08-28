from django.test import TestCase
from django.urls import reverse
from .models import PredictionHistory
from .forms import PredictionForm

class DiabetesModelTests(TestCase):
    def test_record_creation(self):
        record = PredictionHistory.objects.create(
            pregnancies=2,
            glucose=120,
            blood_pressure=80,
            skin_thickness=20,
            insulin=85,
            bmi=25.4,
            diabetes_pedigree_function=0.47,
            age=33,
            predicted_class=0,
            probability=0.245,
            risk_level="LOW"
        )
        self.assertEqual(record.risk_level, "LOW")
        self.assertEqual(record.predicted_class, 0)
        self.assertEqual(record.probability, 0.245)
        self.assertIn("Assessment on", str(record))

class DiabetesFormTests(TestCase):
    def test_valid_form(self):
        data = {
            'pregnancies': 1,
            'glucose': 110,
            'blood_pressure': 75,
            'skin_thickness': 15,
            'insulin': 80,
            'bmi': 24.2,
            'diabetes_pedigree_function': 0.35,
            'age': 28
        }
        form = PredictionForm(data=data)
        self.assertTrue(form.is_valid())

    def test_invalid_negative_value(self):
        data = {
            'pregnancies': 1,
            'glucose': -110,
            'blood_pressure': 75,
            'skin_thickness': 15,
            'insulin': 80,
            'bmi': 24.2,
            'diabetes_pedigree_function': 0.35,
            'age': 28
        }
        form = PredictionForm(data=data)
        self.assertFalse(form.is_valid())
        self.assertIn('glucose', form.errors)

class DiabetesViewTests(TestCase):
    def test_home_view(self):
        response = self.client.get(reverse('diabetes_app:home'))
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'home.html')

    def test_predict_view_get(self):
        response = self.client.get(reverse('diabetes_app:predict'))
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'predict.html')

    def test_about_view(self):
        response = self.client.get(reverse('diabetes_app:about'))
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'about.html')

    def test_analytics_view(self):
        response = self.client.get(reverse('diabetes_app:analytics'))
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'analytics.html')

    def test_history_view(self):
        response = self.client.get(reverse('diabetes_app:history'))
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'history.html')

    def test_prediction_workflow(self):
        # 1. Post valid patient details
        data = {
            'pregnancies': 2,
            'glucose': 148,
            'blood_pressure': 72,
            'skin_thickness': 35,
            'insulin': 0, # zero should be handled
            'bmi': 33.6,
            'diabetes_pedigree_function': 0.627,
            'age': 50
        }
        response = self.client.post(reverse('diabetes_app:predict'), data)
        
        # 2. Assert redirect to results page
        self.assertEqual(response.status_code, 302)
        
        # 3. Check PredictionHistory was created
        self.assertEqual(PredictionHistory.objects.count(), 1)
        record = PredictionHistory.objects.first()
        self.assertEqual(record.glucose, 148.0)
        self.assertEqual(record.age, 50)
        
        # 4. Check result detail page loads correctly
        redirect_url = reverse('diabetes_app:result_detail', kwargs={'pk': record.pk})
        response_result = self.client.get(redirect_url)
        self.assertEqual(response_result.status_code, 200)
        self.assertTemplateUsed(response_result, 'result.html')
        
        # 5. Check variables are present in view context
        self.assertIn('prediction_label', response_result.context)
        self.assertIn('probability_percent', response_result.context)
        self.assertIn('confidence_percent', response_result.context)
        self.assertIn('non_diabetes_probability_percent', response_result.context)
        self.assertIn('gauge_chart', response_result.context)
        
        # 6. Check SHAP explainers are populated
        self.assertIn('explanations', response_result.context)
        explanations = response_result.context['explanations']
        # Sliced to top 5
        self.assertLessEqual(len(explanations), 5)
