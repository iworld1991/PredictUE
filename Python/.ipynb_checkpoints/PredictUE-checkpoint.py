# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.2.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
# #!pip install pytrends
# -

import pandas_datareader.data as web
import datetime
#from pytrends.request import TrendReq
import matplotlib.pyplot as plt
import predictit
import requests
import pandas as pd 
import io
import numpy as np

# ## Import google trends data and michigan data 

# +
## google search 
ue_search = pd.read_excel('../Data/UEGoogle.xls')
ue_search.index = ue_search['Month']

ue_search.plot(lw = 3,
               figsize=(6,4))
# -

## michigan data 
ue_exp = pd.read_excel('../Data/UEExpMichigan.xls',
                       sheet_name = 'Data',
                       index_col = 0)

ue_exp.plot(kind='line',
            figsize = (6,4))

# ## Unemployement rate 

# +
start = datetime.datetime(1960, 1, 30)
end = datetime.datetime(2020, 3, 30)

ue = web.DataReader('UNRATE', 'fred', start, end)
# -

ue.plot(lw = 3)

# ## Combine 

data = pd.concat([interest_over_time_df,pre_data],join = "inner")

data
