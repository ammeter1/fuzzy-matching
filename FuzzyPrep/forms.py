from django import forms

class UploadFileForm(forms.Form):
    topfile = forms.FileField()
    bottomfile = forms.FileField()
    topSheetname = forms.CharField(max_length=50 , required=False)
    bottomSheetname = forms.CharField(max_length=50 , required=False)
    topDelimiter = forms.CharField(max_length=5 , required=False)
    bottomDelimiter = forms.CharField(max_length=5 , required=False)
    topColname = forms.CharField(max_length=50)
    bottomColname = forms.CharField(max_length=50)
    matchstyle = forms.CharField(max_length=50)

class GetFiles(forms.Form):
    sessionId = forms.CharField()
    leftFile = forms.FileField()
    rightFile = forms.FileField()