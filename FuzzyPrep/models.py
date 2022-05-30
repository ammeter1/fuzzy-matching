from django.db import models
from django.forms import ModelForm
# Create your models here.
class Files(models.Model):
    sessionId = models.CharField(max_length=100)
    leftFile = models.FileField()
    rightFile = models.FileField()

class FileForm(ModelForm):
    class Meta:
        model = Files
        fields = ['sessionId', 'leftFile', 'rightFile']