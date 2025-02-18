from django.urls import path
from .views import index, get_answer, submit_feedback

urlpatterns = [
    path('', index, name='index'),
    path('get_answer/', get_answer, name='get_answer'),
    path('submit_feedback/', submit_feedback, name='submit_feedback'),
]