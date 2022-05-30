from django.urls import path
from . import views

app_name = 'fuzzyprep'
urlpatterns = [
    path('', views.index, name='index'),
    path('select',views.selectData, name='selectData'),
    path('upload', views.upload, name='upload'),
    path('step1', views.getSheetnames, name="step1"),
    path('step2', views.getColumnNames, name='step2'),
    path('step4', views.doMatching, name='step4'),
    path('getMemSize', views.getAvailableMemorySize, name="getAvailableMemorySize")
]