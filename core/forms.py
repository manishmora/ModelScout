from django import forms

class DatasetUploadForm(forms.Form):
    name = forms.CharField(label='Dataset Name', max_length=255)
    file = forms.FileField(label='Select File', required=True)
