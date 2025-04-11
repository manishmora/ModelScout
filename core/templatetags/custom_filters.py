from django import template
import pandas as pd

register = template.Library()

@register.filter
def get_item(dictionary, key):
    """Safely gets dictionary item from key in templates."""
    return dictionary.get(key, '-')

@register.simple_tag
def read_column_unique(path, column_name):
    try:
        df = pd.read_csv(path)
        return df[column_name].dropna().unique()
    except Exception:
        return []


# yourapp/templatetags/custom_filters.py

import numpy as np

register = template.Library()

@register.filter
def mean(value):
    try:
        return round(np.mean(value), 4)
    except:
        return "-"

@register.filter
def std(value):
    try:
        return round(np.std(value), 4)
    except:
        return "-"
