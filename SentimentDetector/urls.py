from django.urls import path
# from views import sentiment_detector
from .views import sentiment_detector

urlpatterns = [
    path('', sentiment_detector),
]
