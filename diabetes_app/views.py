from django.shortcuts import render, redirect, get_object_or_404
from django.contrib import messages
from django.core.paginator import Paginator
from django.urls import reverse
import os
import json

from .models import PredictionHistory
from .forms import PredictionForm
from ml.prediction import predict_diabetes_risk
from ml.explainability import explain_prediction_shap

def home(request):
    """
    Polished landing page inspired by a modern AI/web-agency design.
    """
    total_assessments = PredictionHistory.objects.count()
    context = {
        'total_assessments': total_assessments
    }
    return render(request, 'home.html', context)

def predict(request):
    """
    Handles user biometric inputs form page. Runs ML pipeline.
    """
    if request.method == 'POST':
        form = PredictionForm(request.POST)
        if form.is_valid():
            # Build features dictionary
            features = {
                'Pregnancies': form.cleaned_data['pregnancies'],
                'Glucose': form.cleaned_data['glucose'],
                'BloodPressure': form.cleaned_data['blood_pressure'],
                'SkinThickness': form.cleaned_data['skin_thickness'],
                'Insulin': form.cleaned_data['insulin'],
                'BMI': form.cleaned_data['bmi'],
                'DiabetesPedigreeFunction': form.cleaned_data['diabetes_pedigree_function'],
                'Age': form.cleaned_data['age']
            }
            
            try:
                # Predict risk parameters using ML pipeline
                pred_results = predict_diabetes_risk(features)
                predicted_class = pred_results['prediction']
                probability = pred_results['probability']
                risk_level = pred_results['risk_level']
                
                # Save record to SQLite
                record = form.save(commit=False)
                record.predicted_class = predicted_class
                record.probability = probability
                record.risk_level = risk_level
                record.save()
                
                messages.success(request, "Diabetes assessment calculated successfully!")
                return redirect('diabetes_app:result_detail', pk=record.pk)
            except FileNotFoundError as e:
                messages.error(request, f"Prediction system error: {str(e)}")
            except Exception as e:
                messages.error(request, f"An unexpected error occurred during prediction: {str(e)}")
    else:
        form = PredictionForm()
        
    return render(request, 'predict.html', {'form': form})

def result_detail(request, pk):
    """
    Renders risk assessment results, including custom-styled SHAP contribution bars.
    """
    record = get_object_or_404(PredictionHistory, pk=pk)
    
    # Structure features for explainability
    features_dict = {
        'Pregnancies': record.pregnancies,
        'Glucose': record.glucose,
        'BloodPressure': record.blood_pressure,
        'SkinThickness': record.skin_thickness,
        'Insulin': record.insulin,
        'BMI': record.bmi,
        'DiabetesPedigreeFunction': record.diabetes_pedigree_function,
        'Age': record.age
    }
    
    # Calculate SHAP explainers
    shap_results = explain_prediction_shap(features_dict)
    
    # Pre-process explanations for rendering in templates
    # Normalize SHAP values to represent percentage-width values for CSS bars
    explanations = shap_results.get('explanations', [])
    
    if explanations:
        # Limit to the top 5 contributors
        explanations = explanations[:5]
        max_shap = max(abs(item['shap_value']) for item in explanations)
        # Avoid division by zero
        if max_shap == 0:
            max_shap = 1.0
            
        for item in explanations:
            # width_percent dictates CSS progress bar size (relative to highest contributor)
            item['width_percent'] = int((abs(item['shap_value']) / max_shap) * 100)
            # Round SHAP values for clean representation
            item['display_shap'] = round(item['shap_value'], 4)
            
    # Generate Plotly gauge chart
    import plotly.graph_objects as go
    import plotly.io as pio
    
    prob_val = round(record.probability * 100, 1)
    fig_gauge = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = prob_val,
        domain = {'x': [0, 1], 'y': [0, 1]},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "#475569"},
            'bar': {'color': "#ef4444" if record.predicted_class == 1 else "#10b981"},
            'bgcolor': "white",
            'borderwidth': 1,
            'bordercolor': "#cbd5e1",
            'steps': [
                {'range': [0, 33], 'color': 'rgba(16, 185, 129, 0.1)'},
                {'range': [33, 66], 'color': 'rgba(245, 158, 11, 0.1)'},
                {'range': [66, 100], 'color': 'rgba(239, 68, 68, 0.1)'}
            ],
        }
    ))
    fig_gauge.update_layout(
        height=150,
        margin=dict(l=10, r=10, t=10, b=10),
        template='plotly_white'
    )
    gauge_chart = pio.to_html(fig_gauge, full_html=False, include_plotlyjs=False)
            
    context = {
        'record': record,
        'probability_percent': round(record.probability * 100, 1),
        'non_diabetes_probability_percent': round((1.0 - record.probability) * 100, 1),
        'confidence_percent': round(max(record.probability, 1.0 - record.probability) * 100, 1),
        'prediction_label': "Diabetes Detected" if record.predicted_class == 1 else "Diabetes Not Detected",
        'explanations': explanations,
        'shap_success': shap_results.get('success', False),
        'explainer_type': shap_results.get('explainer_type', 'Fallback'),
        'error_msg': shap_results.get('error_msg'),
        'gauge_chart': gauge_chart
    }
    return render(request, 'result.html', context)

def analytics(request):
    """
    Renders metrics from metrics.json using Plotly charts.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    metrics_path = os.path.join(base_dir, 'ml', 'saved_models', 'metrics.json')
    
    metrics_exists = os.path.exists(metrics_path)
    context = {
        'metrics_exists': metrics_exists
    }
    
    if metrics_exists:
        try:
            with open(metrics_path, 'r') as f:
                metrics_data = json.load(f)
                
            best_model_name = metrics_data.get('best_model_name')
            best_model_metrics = metrics_data['models'][best_model_name]['metrics']
            
            context.update({
                'best_model_name': best_model_name,
                'accuracy_percent': round(best_model_metrics['accuracy'] * 100, 1),
                'precision_percent': round(best_model_metrics['precision'] * 100, 1),
                'recall_percent': round(best_model_metrics['recall'] * 100, 1),
                'f1_percent': round(best_model_metrics['f1_score'] * 100, 1),
                'roc_auc_percent': round(best_model_metrics['roc_auc'] * 100, 1),
            })
            
            # Draw Plotly Figures
            import plotly.express as px
            import plotly.graph_objects as go
            import plotly.io as pio
            import pandas as pd
            import numpy as np
            
            # --- 1. Model Comparison Chart ---
            comp_data = []
            for model_name, model_info in metrics_data['models'].items():
                for m_name, m_val in model_info['metrics'].items():
                    comp_data.append({
                        'Model': model_name,
                        'Metric': m_name.replace('_', ' ').title(),
                        'Score': m_val
                    })
            df_comp = pd.DataFrame(comp_data)
            fig_comp = px.bar(
                df_comp, x='Metric', y='Score', color='Model', barmode='group',
                color_discrete_map={'Logistic Regression': '#0d9488', 'Decision Tree': '#0ea5e9', 'Random Forest': '#8b5cf6'}
            )
            fig_comp.update_layout(
                yaxis_range=[0, 1.05], 
                template='plotly_white', 
                height=350,
                font=dict(family='Manrope, sans-serif', size=11),
                margin=dict(l=40, r=20, t=20, b=40),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            chart_comp = pio.to_html(fig_comp, full_html=False, include_plotlyjs=False, config={'responsive': True})
            
            # --- 2. Confusion Matrix Heatmap ---
            cm = metrics_data['models'][best_model_name]['confusion_matrix']
            z = cm
            x_labels = ['Pred Healthy', 'Pred Diabetic']
            y_labels = ['Act Healthy', 'Act Diabetic']
            
            annotations = []
            for i, row in enumerate(z):
                for j, val in enumerate(row):
                    annotations.append(dict(
                        x=x_labels[j], y=y_labels[i], text=str(val),
                        font=dict(color='white' if val > (np.max(z)/2) else 'black', size=16, family='Manrope, Arial'),
                        showarrow=False
                    ))
            fig_cm = go.Figure(data=go.Heatmap(
                z=z, x=x_labels, y=y_labels, colorscale='Blues', showscale=False
            ))
            fig_cm.update_layout(
                annotations=annotations,
                yaxis=dict(autorange="reversed"),
                template='plotly_white',
                height=350,
                font=dict(family='Manrope, sans-serif', size=11),
                margin=dict(l=80, r=40, t=30, b=40)
            )
            chart_cm = pio.to_html(fig_cm, full_html=False, include_plotlyjs=False, config={'responsive': True})
            
            # --- 3. ROC Curve Line Plot ---
            fpr = metrics_data['models'][best_model_name]['roc_curve']['fpr']
            tpr = metrics_data['models'][best_model_name]['roc_curve']['tpr']
            
            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name='ROC Curve', line=dict(color='#ef4444', width=3)))
            fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Chance', line=dict(color='grey', dash='dash')))
            fig_roc.update_layout(
                xaxis_title='False Positive Rate',
                yaxis_title='True Positive Rate',
                xaxis_range=[-0.01, 1.01],
                yaxis_range=[-0.01, 1.01],
                template='plotly_white',
                height=350,
                font=dict(family='Manrope, sans-serif', size=11),
                margin=dict(l=50, r=20, t=20, b=50)
            )
            chart_roc = pio.to_html(fig_roc, full_html=False, include_plotlyjs=False, config={'responsive': True})
            
            # --- 4. Feature Importance Bar Chart ---
            importances = metrics_data['models'][best_model_name]['feature_importances']
            sorted_imp = sorted(importances.items(), key=lambda x: x[1])
            feats = [x[0] for x in sorted_imp]
            vals = [x[1] for x in sorted_imp]
            
            fig_imp = px.bar(
                x=vals, y=feats, orientation='h',
                labels={'x': 'Relative Weight', 'y': 'Feature'},
                color=vals, color_continuous_scale='Viridis'
            )
            fig_imp.update_layout(
                showlegend=False,
                coloraxis_showscale=False,
                template='plotly_white',
                height=350,
                font=dict(family='Manrope, sans-serif', size=11),
                margin=dict(l=150, r=20, t=20, b=40)
            )
            chart_imp = pio.to_html(fig_imp, full_html=False, include_plotlyjs=False, config={'responsive': True})
            
            context.update({
                'chart_comp': chart_comp,
                'chart_cm': chart_cm,
                'chart_roc': chart_roc,
                'chart_imp': chart_imp
            })
            
        except Exception as e:
            context['metrics_exists'] = False
            messages.error(request, f"Failed to generate Plotly figures: {str(e)}")
            
    return render(request, 'analytics.html', context)

def history(request):
    """
    Lists predictions from SQLite database with pagination.
    """
    records = PredictionHistory.objects.all()
    
    paginator = Paginator(records, 10)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)
    
    context = {
        'page_obj': page_obj,
        'total_records': paginator.count
    }
    return render(request, 'history.html', context)

def clear_history(request):
    """
    Clear all prediction histories using POST + CSRF protection.
    """
    if request.method == 'POST':
        PredictionHistory.objects.all().delete()
        messages.success(request, "Prediction assessment history was cleared successfully.")
    else:
        messages.warning(request, "Direct GET access is not allowed for this action.")
    return redirect('diabetes_app:history')

def about(request):
    """
    Methodology walkthrough of the ML model, pipeline flowchart, and details.
    """
    return render(request, 'about.html')
