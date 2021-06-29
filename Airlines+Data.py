# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 18:26:11 2021

@author: DELL
"""

#Importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#loading the dataset
Airlines = pd.read_excel("Airlines+Data.xlsx")

month=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
month=pd.DataFrame(month)
months=pd.DataFrame(np.tile(month,(8,1)))
Airlines=pd.concat([Airlines,months],axis=1)
Airlines.columns=['Month','Passengers','months']


#Creating the dummies
month_dummies=pd.get_dummies(Airlines['months'])
Airlines=pd.concat([Airlines,month_dummies],axis=1)
Airlines['t']=np.arange(1,97)
Airlines['t_sq']=Airlines['t']*Airlines['t']
Airlines['log_passengers']=np.log(Airlines['Passengers'])

#Splitting into Train and Test
Train=Airlines[0:85]
Test=Airlines[85:]
plt.plot(Airlines.iloc[:,1])
Test.set_index(np.arange(1,12),inplace=True)

######## Linear ##########
import statsmodels.formula.api as smf
lin_model=smf.ols('Passengers~t',data=Train).fit()
pred_lin=lin_model.predict(Test['t'])
error_lin=Test['Passengers']-pred_lin
rmse_lin=np.sqrt(np.mean(error_lin**2))
#rmse_lin=55.67

###### Exponential #######
import statsmodels.formula.api as smf
exp_model=smf.ols('log_passengers~t',data=Train).fit()
pred_exp=exp_model.predict(Test['t'])
error_exp=Test['Passengers']-pred_exp
rmse_exp=np.sqrt(np.mean(error_exp**2))
#rmse_exp=329.69

######## Quadratic #######
import statsmodels.formula.api as smf
quad_model=smf.ols('Passengers~t+t_sq',data=Train).fit()
pred_quad=quad_model.predict(Test[['t','t_sq']])
error_quad=Test['Passengers']-pred_quad
rmse_quad=np.sqrt(np.mean(error_quad**2))
#rmse_quad=50.65

####### Additive Seasonality ###########
import statsmodels.formula.api as smf
add_sea_model=smf.ols('Passengers~Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data=Train).fit()
pred_add_sea_model=add_sea_model.predict(Test[['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov']])
error_add_sea_model=Test['Passengers']-pred_add_sea_model
rmse_add_sea_model=np.sqrt(np.mean(error_add_sea_model**2))
#rmse_add_sea_model=134.34

######### Additive Seasonality Quadratic #########
import statsmodels.formula.api as smf
add_sea_quad=smf.ols('Passengers~Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov+t+t_sq',data=Train).fit()
pred_add_sea_quad=add_sea_quad.predict(Test[['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','t','t_sq']])
error_add_sea_quad=Test['Passengers']-pred_add_sea_quad
rmse_add_sea_quad=np.sqrt(np.mean(error_add_sea_quad**2))
#rmse_add_sea_quad=27.41

######## Multiplicative seasonality ########
import statsmodels.formula.api as smf
mul_sea_model=smf.ols('log_passengers~Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data=Train).fit()
pred_mul_sea_model=mul_sea_model.predict(Test[['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov']])
error_mul_sea_model=Test['Passengers']-pred_mul_sea_model
rmse_mul_sea=np.sqrt(np.mean(error_mul_sea_model**2))
#rmse_mul_sea=330.19

####### Multiplicative Additive Seasonality ############
import statsmodels.formula.api as smf
mul_add_sea=smf.ols('log_passengers~t+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data=Train).fit()
pred_mul_add=mul_add_sea.predict(Test[['t','Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov']])
error_mul_add=Test['Passengers']-pred_mul_add
rmse_mul_add=np.sqrt(np.mean(error_mul_add**2))
#rmse_mul_add=329.66

#Testing
data={'Model':['rmse_lin','rmse_exp','rmse_quad','rmse_add_sea_model','rmse_add_sea_quad','rmse_mul_sea','rmse_mul_add'],
      'rmse_val':[rmse_lin,rmse_exp,rmse_quad,rmse_add_sea_model,rmse_add_sea_quad,rmse_mul_sea,rmse_mul_add]}
table_rmse=pd.DataFrame(data)
table_rmse

#So rmse_add_sea_quad has the least value(27.41) among all the models so far
#So, Additive seasonality Quadratic model is the best model.















