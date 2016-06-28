# Data Management & Visualization Class
# AddHealth research project

# Capstone project data analysis python script.

# Author: Jeff Spoelstra

# Import required libraries
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing as skp
from sklearn.ensemble import RandomForestClassifier
import sklearn.metrics as skm
import numpy as np
import matplotlib.pyplot as plt



# bug fix for display formats to avoid run time errors
pd.set_option('display.float_format', lambda x:'%f'%x)

# Load the waterpoints data set. The data variables are stored in a "values"
# file and the functional labels stored in a "labels" file.
# NOTE: the data files must be located in the same working directory as this
# script file.
rdatav = pd.read_csv(".\\Training_set_values.csv", low_memory=False)
rdatal = pd.read_csv(".\\Training_set_labels.csv", low_memory=False)

# Join the input data using 'id' column as a key.
rdatav.set_index(keys='id', inplace=True, drop=False)
rdatal.set_index(keys='id', inplace=True)

trn_rdata = pd.concat([rdatal, rdatav], axis=1)

#trn_rdata.reset_index(inplace=True, drop=False)
del rdatav, rdatal

# Separate out just the variables we are going to work with.
# NOTE: the copy() method is needed to stop pandas from throwing
# spurious warnings.
pred = trn_rdata[['id', 'region_code', 'district_code', 'ward',
                  'installer', 'funder', 'permit', 'construction_year',
                  'management_group',
                  'extraction_type_group', 'waterpoint_type_group','quantity_group',
                  'basin', 'source_type', 'quality_group']].copy()
resp = trn_rdata[['status_group']].copy()

# Clean up NA data.
pred['ward'] = pred['ward'].replace(np.nan, ' ')
#pred['subvillage'] = pred['subvillage'].replace(np.nan, ' ')
pred['installer'] = pred['installer'].replace(np.nan, ' ')
pred['funder'] = pred['funder'].replace(np.nan, ' ')
pred['permit'] = pred['permit'].replace(np.nan, False)
pred['construction_year'] = pred['construction_year'].replace(np.nan, 0)
pred['management_group'] = pred['management_group'].replace(np.nan, ' ')
pred['extraction_type_group'] = pred['extraction_type_group'].replace(np.nan, ' ')
pred['waterpoint_type_group'] = pred['waterpoint_type_group'].replace(np.nan, ' ')
pred['quantity_group'] = pred['quantity_group'].replace(np.nan, ' ')
pred['basin'] = pred['basin'].replace(np.nan, ' ')
pred['source_type'] = pred['source_type'].replace(np.nan, ' ')
pred['quality_group'] = pred['quality_group'].replace(np.nan, ' ')

# Convert all categorical data to numerical equivalents.
for col in pred.columns :
    if(pred[col].dtype == 'object') :
        pred[col] = skp.LabelEncoder().fit_transform(pred[col])




##########################################
##########################################
#
# In this first pass we are just going to sort out 'functional' vs
# 'non functional' waterpoints; for now treating 'functional needs repair'
# as 'functional'.
#
##########################################
##########################################

# Remap the operation status values to smaller strings for convenience.
recode1 = {'functional': 'FUNC', 
           'functional needs repair': 'FUNC',
           'non functional': 'NONFUNC'}
resp = resp.replace(recode1)

# Randomly split the raw data into a working training set and a validation set.
pred_train, pred_val, resp_train, resp_val = train_test_split(pred, 
                                                              resp, 
                                                              test_size=.3)
# Create a random forest classifier using 100 trees.
classifier = RandomForestClassifier(n_estimators=100, oob_score=True)

# Fit a series of models using each predictor variable invidually to see which
# ones may have the greatest impact on overall accuracy.
# Along the way, keep track of the variables and their accuracy rates.
collist = list()
errlist = list()
for col in pred_train.columns[1:] :         # skip waterpoint id column
    print('Evaluating [', col, ']')
    pvals = pred_train.as_matrix([col])
    classifier = classifier.fit(pvals, resp_train.status_group)
    collist.append(col)
    errlist.append(classifier.oob_score_)

# Print the accuracy rates in order of descending accuracy rate.
df = pd.DataFrame(data={'Accuracy': errlist}, index=collist)
df.sort_values('Accuracy', ascending=False, inplace=True)
print('Accuracy rates for single-predictor models (FUNC vs NONFUNC):')
print(df)

# Fit a series of models that one-at-time add the predictor variables in
# descending order from highest individual accuracy to lowest.
# Along the way, keep track of the predictor count and the accuracy rates.
pred2 = pred_train[df.index]
collist = list()
errlist = list()

for colidx in range(1, len(df)+1) :
    pvals = pred2.ix[:,0:colidx]    
    print(pvals.shape)
    classifier = classifier.fit(pvals, resp_train.status_group)
    collist.append(pvals.shape[1])
    errlist.append(classifier.oob_score_)

# Print the accuracy rates in order of descending accuracy rate.
df2 = pd.DataFrame(data={'Predictors': collist, 'Accuracy': errlist},
                   columns=['Predictors', 'Accuracy'])
df2.sort_values('Predictors', ascending=False, inplace=True)
print('Accuracy rates for multi-predictor models (FUNC v NONFUNC):')
print(df2)

# Plot the error rate vs number of predictors in the model.
df2.sort_values('Predictors', inplace=True)
fig = plt.figure(figsize=(6, 5))
plt.plot(df2.Predictors, (1 - df2.Accuracy), linewidth=2.0)
plt.title('Figure 2: Estimated Error Rate', size=14)
plt.xlabel('Number of Predictors')
plt.ylabel('Estimated OOB Error Rate')
plt.savefig('Figure 2.png')
plt.show()




## Re-do the stepwise adding of predictors, but this time leave out the subvillage
## variable which appears to have a depressive affect on overall accuracy.
#df3 = df.drop('subvillage')
#
#pred3 = pred_train[df3.index]
#collist = list()
#errlist = list()
#
#for colidx in range(1, len(df3)+1) :
#    pvals = pred3.ix[:,0:colidx]    
#    print(pvals.shape)
#    classifier = classifier.fit(pvals, resp_train.status_group)
#    collist.append(pvals.shape[1])
#    errlist.append(classifier.oob_score_)
#
## Print the new accuracy rates in order of descending accuracy rate.
#df4 = pd.DataFrame(data={'Predictors': collist, 'Accuracy': errlist},
#                   columns=['Predictors', 'Accuracy'])
#df4.sort_values('Predictors', ascending=False, inplace=True)
#print('Accuracy rates of multi-predictor models w/o subvillage variable:')
#print(df4)
#
## Plot the accuracy rate vs number of predictors in the model.
#df4.sort_values('Predictors', inplace=True)
#
#fig = plt.figure(figsize=(6, 5))
#plt.plot(df2.Predictors, (1 - df2.Accuracy), 'b',
#         df4.Predictors, (1 - df4.Accuracy), 'g', 
#         linewidth=2.0)
#plt.title('Figure 2: Estimated Error Rate', size=14)
#plt.xlabel('Number of Predictors')
#plt.ylabel('Estimated OOB Error Rate')
#plt.legend(['With village', 'Without village'])
#plt.savefig('Figure 2.png')
#plt.show()

#%%

# Train a model with just the top 10 predictors. More predictors than that does
# not appear produce significant improvement in accuracy rate.
#final_vars = df3.index[0:10]
final_vars = df2.index[0:10]
final_pred = pred_train[final_vars]

classifier_FvNF = classifier.fit(final_pred, resp_train.status_group)

print('Final predictor set estimated OOB accuracy (FUNC vs NONFUNC):', classifier_FvNF.oob_score_)

# Use the final fitted model to predict waterpoint operation status in the
# validation set using only the chosen predictors.
predictions_FvNF = classifier_FvNF.predict(pred_val[final_vars])

# Show a confusion matrix and accuracy score.
ct = pd.crosstab(resp_val.status_group, predictions_FvNF, rownames=['True'], 
                 colnames=['Predicted'], margins=True)
print(ct)
print('Estimated OOB accuracy from training (FUNC vs NONFUNC): ', classifier_FvNF.oob_score_)
print('Validation set accuracy (FUNC vs NONFUNC): ', skm.accuracy_score(resp_val, predictions_FvNF))

# Make a nice looking chart of the confusion matrix
colLabels = list(ct.columns.values)
rowLabels = list(ct.index)
#nrows, ncols = len(ct) + 1, len(colLabels) + 1
#hcell, wcell = 0.5, 1.
#hpad, wpad = 0, 0    

#fig = plt.figure(figsize=(ncols*wcell+wpad, nrows*hcell+hpad))
fig = plt.figure(figsize=(6, 5))
ax = plt.gca()
#ax.xaxis.set_visible(False)
#ax.yaxis.set_visible(False)
ax.axis('off')

cellText=[]
for row in range(0, len(ct)) :
    cellText.append(list(ct.iloc[row,:]))
    
the_table = plt.table(cellText=cellText,
          rowLabels=rowLabels,
          colLabels=colLabels,
          loc='center')
plt.title('Figure 3: Prediction Confusion Matrix')
plt.savefig('Figure 3.png')
plt.show()

#%%

##########################################
##########################################
#
# Now do it all over again but we're going to just separate 'functional'
# from 'functional needs repair' water points by removing 'non functional'
# water points from the training data and fitting a new model.
#
##########################################
##########################################

# Filter rdata and retain just the 'functional' and 'function needs repair'
# records.
ffnr_rdata = trn_rdata[trn_rdata['status_group'].isin(['functional', 
                                                       'functional needs repair'])]

# Separate out just the variables we are going to work with.
# NOTE: the copy() method is needed to stop pandas from throwing
# spurious warnings.
pred = ffnr_rdata[['id', 'region_code', 'district_code', 'ward',
                   'installer', 'funder', 'permit', 'construction_year',
                   'management_group',
                   'extraction_type_group', 'waterpoint_type_group','quantity_group',
                   'basin', 'source_type', 'quality_group']].copy()
resp = ffnr_rdata[['status_group']].copy()

# Clean up NA data.
pred['ward'] = pred['ward'].replace(np.nan, ' ')
#pred['subvillage'] = pred['subvillage'].replace(np.nan, ' ')
pred['installer'] = pred['installer'].replace(np.nan, ' ')
pred['funder'] = pred['funder'].replace(np.nan, ' ')
pred['permit'] = pred['permit'].replace(np.nan, False)
pred['construction_year'] = pred['construction_year'].replace(np.nan, 0)
pred['management_group'] = pred['management_group'].replace(np.nan, ' ')
pred['extraction_type_group'] = pred['extraction_type_group'].replace(np.nan, ' ')
pred['waterpoint_type_group'] = pred['waterpoint_type_group'].replace(np.nan, ' ')
pred['quantity_group'] = pred['quantity_group'].replace(np.nan, ' ')
pred['basin'] = pred['basin'].replace(np.nan, ' ')
pred['source_type'] = pred['source_type'].replace(np.nan, ' ')
pred['quality_group'] = pred['quality_group'].replace(np.nan, ' ')

# Convert all categorical data to numerical equivalents.
for col in pred.columns :
    if(pred[col].dtype == 'object') :
        pred[col] = skp.LabelEncoder().fit_transform(pred[col])

# Remap the operation status values to smaller strings for convenience.
recode1 = {'functional': 'FUNC', 
           'functional needs repair': 'FUNCNR'}
resp = resp.replace(recode1)

# Randomly split the raw data into a workikng training set and a validation set.
pred_train, pred_val, resp_train, resp_val = train_test_split(pred, 
                                                              resp, 
                                                              test_size=.3)

#%%


# Create a random forest classifier using 100 trees.
classifier = RandomForestClassifier(n_estimators=100, oob_score=True)

# Fit a series of models using each predictor variable invidually to see which
# ones may have the greatest impact on overall accuracy.
# Along the way, keep track of the variables and their accuracy rates.
collist = list()
errlist = list()
for col in pred_train.columns[1:] :         # skip waterpoint id column
    print('Evaluating [', col, ']')
    pvals = pred_train.as_matrix([col])
    classifier = classifier.fit(pvals, resp_train.status_group)
    collist.append(col)
    errlist.append(classifier.oob_score_)

# Print the accuracy rates in order of descending accuracy rate.
df = pd.DataFrame(data={'Accuracy': errlist}, index=collist)
df.sort_values('Accuracy', ascending=False, inplace=True)
print('Accuracy rates for single-predictor models (FUNC vs FUNCNR):')
print(df)

# Fit a series of models that one-at-time add the predictor variables in
# descending order from highest individual accuracy to lowest.
# Along the way, keep track of the predictor count and the accuracy rates.
pred2 = pred_train[df.index]
collist = list()
errlist = list()

for colidx in range(1, len(df)+1) :
    pvals = pred2.ix[:,0:colidx]    
    print(pvals.shape)
    classifier = classifier.fit(pvals, resp_train.status_group)
    collist.append(pvals.shape[1])
    errlist.append(classifier.oob_score_)

# Print the accuracy rates in order of descending accuracy rate.
df2 = pd.DataFrame(data={'Predictors': collist, 'Accuracy': errlist},
                   columns=['Predictors', 'Accuracy'])
df2.sort_values('Predictors', ascending=False, inplace=True)
print('Accuracy rates for multi-predictor models (FUNC vs FUNCNR):')
print(df2)

# Plot the accuracy rate vs number of predictors in the model.
df2.sort_values('Predictors', inplace=True)
plt.plot(df2.Predictors, (1 - df2.Accuracy))
plt.show()

#%%

# Re-do the stepwise adding of predictors, but this time leave out the
# variables which appear to have a depressive affect on overall accuracy.
df3 = df.drop(['region_code', 'management_group', 'quantity_group',
               'quality_group', 'subvillage', 'permit', 'basin', 'installer',
               'source_type', 'construction_year'])

pred3 = pred_train[df3.index]
collist = list()
errlist = list()

for colidx in range(1, len(df3)+1) :
    pvals = pred3.ix[:,0:colidx]    
    print(pvals.shape)
    classifier = classifier.fit(pvals, resp_train.status_group)
    collist.append(pvals.shape[1])
    errlist.append(classifier.oob_score_)

# Print the new accuracy rates in order of descending accuracy rate.
df4 = pd.DataFrame(data={'Predictors': collist, 'Accuracy': errlist},
                   columns=['Predictors', 'Accuracy'])
df4.sort_values('Predictors', ascending=False, inplace=True)
print('Accuracy rates of multi-predictor models w/o confounding variables:')
print(df4)

# Plot the accuracy rate vs number of predictors in the model.
df4.sort_values('Predictors', inplace=True)

fig = plt.figure(figsize=(6, 5))
plt.plot(df2.Predictors, (1 - df2.Accuracy), 'b',
         df4.Predictors, (1 - df4.Accuracy), 'g', 
         linewidth=2.0)
plt.title('Figure 4: Estimated Error Rate', size=14)
plt.xlabel('Number of Predictors')
plt.ylabel('Estimated OOB Error Rate')
plt.legend(['All', 'W/O Confounders'])
plt.savefig('Figure 4.png')
plt.show()


#%%

# Train a model with just the remaining predictors as they all appear to be 
# significant.
final_vars = df3.index
final_pred = pred_train[final_vars]

classifier = classifier.fit(final_pred, resp_train.status_group)

print('Final predictor set estimated OOB accuracy (FUNC vs FUNCNR):', classifier.oob_score_)

# Use the final fitted model to predict waterpoint operation status in the
# validation set using only the chosen predictors.
predictions = classifier.predict(pred_val[final_vars])

# Show a confusion matrix and accuracy score.
ct = pd.crosstab(resp_val.status_group, predictions, rownames=['True'], 
                 colnames=['Predicted'], margins=True)
print(ct)
print('Estimated OOB accuracy from training (FUNC vs FUNCNR): ', classifier.oob_score_)
print('Validation set accuracy (FUNC vs FUNCNR): ', skm.accuracy_score(resp_val, predictions))

# Make a nice looking chart of the confusion matrix
colLabels = list(ct.columns.values)
rowLabels = list(ct.index)
#nrows, ncols = len(ct) + 1, len(colLabels) + 1
#hcell, wcell = 0.5, 1.
#hpad, wpad = 0, 0    

#fig = plt.figure(figsize=(ncols*wcell+wpad, nrows*hcell+hpad))
fig = plt.figure(figsize=(6, 5))
ax = plt.gca()
#ax.xaxis.set_visible(False)
#ax.yaxis.set_visible(False)
ax.axis('off')

cellText=[]
for row in range(0, len(ct)) :
    cellText.append(list(ct.iloc[row,:]))
    
the_table = plt.table(cellText=cellText,
          rowLabels=rowLabels,
          colLabels=colLabels,
          loc='center')
plt.title('Figure 5: Prediction Confusion Matrix')
plt.savefig('Figure 5.png')
plt.show()

#%%

# *************************************************
# *************************************************
# This script will now test the final prediction model using the true test 
# data set.
#
# Read and pre-process the test data in the same manner as was done with the
# training data. As with the training data set, the files must be located in
# the same working directory as this script.
rdata = pd.read_csv(".\\Test_set_values.csv", low_memory=False)

# Separate out just the final set of variables we are going to work with.
# NOTE: the copy() method is needed to stop pandas from throwing
# spurious warnings.
test = rdata[final_vars].copy()

test['ward'] = test['ward'].replace(np.nan, ' ')
test['installer'] = test['installer'].replace(np.nan, ' ')
test['funder'] = test['funder'].replace(np.nan, ' ')
test['construction_year'] = test['construction_year'].replace(np.nan, 0)
test['extraction_type_group'] = test['extraction_type_group'].replace(np.nan, ' ')
test['waterpoint_type_group'] = test['waterpoint_type_group'].replace(np.nan, ' ')
test['quantity_group'] = test['quantity_group'].replace(np.nan, ' ')

# Convert all categorical data to numerical equivalents.
for col in test.columns :
    if(test[col].dtype == 'object') :
        test[col] = skp.LabelEncoder().fit_transform(test[col])

# Use the final fitted model to predict waterpoint operation status in the
# test set.
predictions = classifier.predict(test)

# Remap the operation status values to the correct strings.
pdf = pd.DataFrame(data={'Predictions': predictions})
recode1 = {'FUNC': 'functional', 
           'FUNCNR': 'functional needs repair',
           'NONFUNC': 'non functional'}
pdf = pdf.replace(recode1)



