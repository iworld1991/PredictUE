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

import pandas_datareader.data as web
import datetime
#from pytrends.request import TrendReq
import matplotlib.pyplot as plt
import predictit
import requests
import pandas as pd 
import io
import numpy as np
import statsmodels.api as sm
from IPython.display import display, Markdown, Latex


figsize = (8,5)
fontsize = 10

# ## Import google trends data and Michigan data 

# + {"code_folding": []}
## google search 
ue_search = pd.read_excel('../Data/UEGoogle.xls')
ue_search.index = ue_search['Month']

ue_search.index = pd.DatetimeIndex(pd.to_datetime(ue_search.index,
                                                  format = '%Y-%m'),
                                  freq='infer')
ue_search.index.name = None

ue_search = ue_search.rename(columns = {'unemployment insurance: (United States)':'Search: \"unemployment insurance\"',
                             'unemployment office: (United States)':'Search: \"unemployment office\"',
                             'file for unemployment: (United States)':'Search: \"file for unemployment\"'})


# +
searches = ['Search: \"unemployment insurance\"',
                'Search: \"unemployment office\"',
                'Search: \"file for unemployment\"']

##########################################################
sub_searches = searches[0:2]
#########################################################

ue_search = ue_search[sub_searches]

# + {"code_folding": []}
ue_search.plot(lw = 3,
               figsize = figsize,
               title = 'google searches of unemployment-related words')
plt.savefig('figures/search')

# + {"code_folding": []}
## normalize each indicies by its initial value. 

for search in sub_searches:
    ue_search[search] = ue_search[search]*100/ue_search[search][0]
# -

## after normalization
ue_search.plot(lw = 3,
               figsize = figsize,
               title = 'google searches of unemployment-related words (normalized)')
plt.savefig('figures/search_normalized')


# +
## michigan data 
ue_exp = pd.read_excel('../Data/UEExpMichigan.xls',
                       sheet_name = 'Data',
                       index_col = 0)

ue_exp = ue_exp.loc[ue_exp.index.dropna(how='all')]

ue_exp.index = pd.DatetimeIndex(pd.to_datetime(ue_exp.index,
                                               format = '%Y-%m-%d'),
                                  freq = 'infer')
ue_exp.index.name = None
# -

ue_exp.plot(lw = 3,
            figsize = figsize,
           title = 'unemployment expectation index')
plt.savefig('figures/ue_exp_idx')

ue_exp.tail()

# ## Unemployement rate 

# +
start = datetime.datetime(1960, 1, 30)
end = datetime.datetime(2020, 3, 30)

ue = web.DataReader('UNRATE', 'fred', start, end)
# -

ue.plot(lw = 3,
        figsize = figsize,
        title = 'unemployment rate')
plt.savefig('figures/ue')

ue.index = pd.DatetimeIndex(pd.to_datetime(ue_exp.index,
                                               format = '%Y-%m-%d'),
                                  freq = 'infer')
ue.index.name = None

# ## Combine 

# +
temp = pd.merge(ue_search,
                ue_exp,
                left_index = True,
                right_index = True,
                how = 'outer'
               )

uedf = pd.merge(temp,
               ue,
               left_index = True,
               right_index = True,
                how = 'outer')
# -

uedf.columns

# +
fig, ax = plt.subplots(figsize = figsize)
ax2 = ax.twinx()
ax.plot(uedf.index,uedf['UNRATE'],lw =2, label = 'unemployment rate')
ax2.plot(uedf.index,uedf['UMEX_R'],'r--',lw = 2, label = 'unemployment expectation index')
ax2.plot(uedf.index,uedf['Search: \"unemployment insurance\"'],'k-.',lw = 2, label = 'google search: unemployment insurance')
ax2.plot(uedf.index,uedf['Search: \"unemployment office\"'],'g-.',lw = 1, label = 'google search: unemployment office')
#ax2.plot(uedf.index,uedf['Search: \"file for unemployment\"'],'g-.',lw = 1, label = 'google search: file for unemployment')
ax.set_xlabel("month",fontsize = fontsize)
ax.set_ylabel('%',fontsize = fontsize)
ax2.set_ylabel('index (%)',fontsize = fontsize)

ax.legend(loc = 0,
          fontsize = fontsize)
ax2.legend(loc = 2,
          fontsize = fontsize)
plt.savefig('figures/all')
# -
# ## Regression


# ### Step 1.  predict michigan index using google search  
#
#
# \begin{eqnarray}
# \underbrace{UEI_{t}}_{\text{Unemployment expectation index}} = \alpha + \sum^3_{k=1}\beta_k \text{Search}_{k,t} + \epsilon_{t}
# \end{eqnarray}
#
# - $UEI$: unemployment expectation index
# - $Search_{k,t}$: google search index for querry $k$, e.g. "unemployment insurance", "unemployment office", "file for unemployment", etc. Note that since each google search query taken from Google trends is normalized by its own historical peak value, the level of the index can not be directly comparable across different queries. All indicies should be normalized by their initial value at the first period of the sample. 

uedf = uedf.rename(columns = {'UNRATE':'ue',
                              'UMEX_R':'ue_exp_idx'})

# +
vars_reg = sub_searches + ['ue_exp_idx']

uedf_short1 = uedf[vars_reg].dropna(how ='any')

Y = uedf_short1[['ue_exp_idx']]
X = uedf_short1[sub_searches]
X = sm.add_constant(X)
model = sm.OLS(Y,X)
results = model.fit()
print(results.summary())
# -

fig = plt.figure(figsize = figsize)
plt.plot(uedf_short1.index,
         uedf_short1['ue_exp_idx'], lw = 2, label = 'realized')
plt.plot(uedf_short1.index,
         results.predict(),'r--',lw = 2,label='prediction')
plt.title('Predicting unemployment expectation using google searches')
plt.legend(loc = 2)
plt.savefig('figures/ue_exp_idx_predict')

print('Estimated coefficients are')
coefs1 = results.params
print(coefs1)

uedf.columns

# ### Step 2.  predict future realized unemployment rate change using expectations 

# \begin{eqnarray}
# U_{t+h} - U_{t+h-1} = u + UEI_{t} + \eta_{t+2}
# \end{eqnarray}
#
# - $U_{t+h}$: h-month-ahead realized unemployment rate, h = 1 by default. change h to predict for different horizons 
# - $UEI_{t}$: unemployment rate expectation index at time $t$ 

# +
uedf['ue_chg'] = uedf['ue'].diff(periods = 1) ## monthly change of unemployment rate 
uedf_short2 = uedf[['ue_chg','ue_exp_idx']].dropna(how ='any')

## # of months lag 
############################################################
h = 1  #by default, next month unemployment rate 
#############################################################

Y = np.array(uedf_short2['ue_chg'][h:])
X = np.array(uedf_short2['ue_exp_idx'][:-h])
X = sm.add_constant(X)
model2 = sm.OLS(Y,X)
results2 = model2.fit()
print(results2.summary())
# -

coefs2 = results2.params
print('Estimated coefficients are')
coefs2

fig = plt.figure(figsize = figsize)
plt.plot(uedf_short2.index[h:],
         np.array(uedf_short2['ue_chg'][h:]), 
         '--',
         lw = 2, 
         label = 'realized')
plt.plot(uedf_short2.index[:-h],
         results2.predict(),
         'r-',
         lw = 2,
         label='prediction')
plt.title('Predicting unemployment changes using expectations')
plt.legend(loc = 2)
plt.savefig('figures/ue_change_predict')

# + {"code_folding": []}
## 2-step procedure to predict the unemployment rate change in March 2020 and onward

# + {"code_folding": []}
searches = ['Search: \"unemployment insurance\"',
            'Search: \"unemployment office\"',
            'Search: \"file for unemployment\"']

## predict unemployment exp index 
ue_exp_idx_predicted = coefs1[0] + (coefs1[1]*uedf[searches[0]]
                                   + coefs1[2]*uedf[searches[1]]
                                  # + coefs1[3]*uedf[searches[2]]
                                  )
# -

ue_exp_idx_predicted.tail().plot(title = 'predicted unemployment expectation index')
plt.savefig('figures/ue_exp_idx_predict_recent')

## predict unemployment changes 
ue_ch_predicted = coefs2[0] + coefs2[1]*ue_exp_idx_predicted

# +
## change in unemployment rate 

ue_ch_predicted.tail()
# -

ue_ch_predicted.tail().plot(title = 'predicted change in unemployment rate (in percentage points)')
plt.savefig('figures/ue_change_predict_recent')


