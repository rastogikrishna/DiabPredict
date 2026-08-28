from django.urls import path
from . import views

app_name = 'diabetes_app'

urlpatterns = [
    path('', views.home, name='home'),
    path('predict/', views.predict, name='predict'),
    path('result/<int:pk>/', views.result_detail, name='result_detail'),
    path('analytics/', views.analytics, name='analytics'),
    path('history/', views.history, name='history'),
    path('history/clear/', views.clear_history, name='clear_history'),
    path('about/', views.about, name='about'),
]
