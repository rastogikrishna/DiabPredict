from django.contrib import admin
from .models import PredictionHistory

@admin.register(PredictionHistory)
class PredictionHistoryAdmin(admin.ModelAdmin):
    list_display = ('timestamp', 'glucose', 'bmi', 'age', 'risk_level', 'probability')
    list_filter = ('risk_level', 'timestamp')
    search_fields = ('risk_level',)
    readonly_fields = ('timestamp', 'predicted_class', 'probability', 'risk_level')
    ordering = ('-timestamp',)
