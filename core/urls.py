from django.urls import path
from . import views

urlpatterns = [
    path('', views.dataset_list, name='dataset_list'),  # Landing page
    path('datasets/', views.dataset_list, name='dataset_list'),
    path('upload/', views.upload_dataset, name='upload_dataset'),
    path('dataset/<int:dataset_id>/', views.dataset_details, name='dataset_details'),
    path('dataset/<int:dataset_id>/train/<str:algo_type>/', views.train_model, name='train_model'),
    path('dataset/delete/<int:pk>/', views.delete_dataset, name='delete_dataset'),
    
    # path('dataset/<int:dataset_id>/predict/', views.predict_user_input, name='predict_user_input'),
    # path('media/<path:path>', views.serve_media, name='serve_media'),
]
