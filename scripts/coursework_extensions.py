"""Coursework extensions: geospatial demand, preprocessing audit and stockout prediction.

Run after pipeline.py; all outputs are based on the supplied FMCG sales file.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from pipeline import ROOT, load_sales

CATEGORICAL=['sku_id','store_id','channel','country']
NUMERIC=['day_of_week','month','is_weekend','is_holiday','lag_stockout_1','lag_stockout_7mean','lag_units_1','lag_units_7mean']


def main(zip_path:Path):
    reports=ROOT/'reports';fig=reports/'figures';fig.mkdir(parents=True,exist_ok=True)
    # Required fields only: load and process without merging unrelated CSV files.
    raw=load_sales(zip_path)
    locations=raw.groupby(['country','city','store_id','latitude','longitude'],as_index=False).agg(
        units_sold=('units_sold','sum'),net_sales=('net_sales','sum'),
        stockout_rate=('stock_out_flag','mean'),lead_time_mean=('lead_time_days','mean'),
        lead_time_sd=('lead_time_days','std'))
    locations['lead_time_sd']=locations.lead_time_sd.fillna(0)
    locations.to_csv(reports/'geospatial_sales_density.csv',index=False)
    # City aggregation prevents overlapping labels when multiple stores share a city.
    cities=locations.groupby(['country','city'],as_index=False).agg(
        latitude=('latitude','mean'),longitude=('longitude','mean'),
        units_sold=('units_sold','sum'),store_count=('store_id','nunique'))
    fig_,ax=plt.subplots(figsize=(10,6))
    size=55+270*(cities.units_sold/cities.units_sold.max())
    image=ax.scatter(cities.longitude,cities.latitude,c=cities.units_sold,s=size,
                     cmap='viridis',alpha=.8,edgecolor='black',linewidth=.4)
    for _,r in cities.iterrows():
        ax.annotate(r.city,(r.longitude,r.latitude),xytext=(6,7),
                    textcoords='offset points',fontsize=8)
    fig_.colorbar(image,ax=ax,label='Recorded units sold (city total)')
    ax.set(xlabel='Longitude (degrees)',ylabel='Latitude (degrees)',
           title='Geospatial demand: city totals across 13 stores (2021–2023)')
    ax.grid(alpha=.2);fig_.tight_layout();fig_.savefig(fig/'geospatial_demand.png',dpi=155);plt.close(fig_)

    # Definition of logistics hotspots is a transparent relative ranking, not a geographic accessibility claim.
    locations['high_demand']=locations.units_sold>=locations.units_sold.quantile(.75)
    locations['high_variability']=locations.lead_time_sd>=locations.lead_time_sd.quantile(.75)
    locations['logistics_hotspot']=locations.high_demand & locations.high_variability
    locations.to_csv(reports/'geospatial_logistics_hotspots.csv',index=False)

    # Historical features are generated *before* the chronological split.
    # Each row represents a store-SKU-date observation; recorded sale days need not be contiguous.
    data=raw[['date','sku_id','store_id','channel','country','units_sold','stock_out_flag',
              'is_weekend','is_holiday']].copy()
    del raw
    data=data.sort_values(['store_id','sku_id','date'])
    groups=data.groupby(['store_id','sku_id'],sort=False,observed=True)
    for source,target in [('stock_out_flag','lag_stockout_1'),('units_sold','lag_units_1')]:
        data[target]=groups[source].shift(1)
    data['lag_stockout_7mean']=groups['stock_out_flag'].transform(lambda s:s.shift(1).rolling(7,min_periods=3).mean())
    data['lag_units_7mean']=groups['units_sold'].transform(lambda s:s.shift(1).rolling(7,min_periods=3).mean())
    data['day_of_week']=data.date.dt.dayofweek
    data['month']=data.date.dt.month
    data=data.dropna(subset=['lag_stockout_1','lag_units_1','lag_stockout_7mean','lag_units_7mean'])
    split=pd.Timestamp('2023-07-01')
    training=data[data.date<split]
    testing=data[data.date>=split]
    # A bounded, reproducible sample of training observations keeps the classifier runnable on a laptop.
    # All test observations are preserved.
    if len(training)>230000:
        training=training.sample(n=230000,random_state=42).sort_values('date')
    preprocess=ColumnTransformer([
        ('numeric',Pipeline([('imputer',SimpleImputer(strategy='median')),('scaler',StandardScaler())]),NUMERIC),
        ('categorical',OneHotEncoder(handle_unknown='ignore',sparse_output=True),CATEGORICAL)
    ])
    # Fitted on TRAIN only; transform test with same fitted category vocabulary and scaler statistics.
    model=Pipeline([('preprocess',preprocess),
                    ('classifier',LogisticRegression(max_iter=300,class_weight='balanced',random_state=42))])
    model.fit(training[NUMERIC+CATEGORICAL],training.stock_out_flag.astype(int))
    probabilities=model.predict_proba(testing[NUMERIC+CATEGORICAL])[:,1]
    # Threshold is fixed in advance; not optimized on the test set.
    predictions=(probabilities>=.5).astype(int)
    y=testing.stock_out_flag.astype(int).to_numpy()
    baseline=np.zeros_like(y)
    rows=[]
    for name,p in [('Always predict no stockout',baseline),('Logistic Regression (threshold 0.50)',predictions)]:
        rows.append({'model':name,'test_rows':int(len(y)),'test_prevalence':float(y.mean()),
                     'precision':float(precision_score(y,p,zero_division=0)),
                     'recall':float(recall_score(y,p,zero_division=0)),
                     'f1':float(f1_score(y,p,zero_division=0)),
                     'predicted_positive_rate':float(np.mean(p))})
    metrics=pd.DataFrame(rows)
    metrics.to_csv(reports/'stockout_classification_metrics.csv',index=False)
    by_channel=pd.DataFrame({'channel':testing.channel.to_numpy(),'actual':y,'predicted':predictions}).groupby('channel').apply(
        lambda grp:pd.Series({'records':len(grp),'stockout_rate':grp.actual.mean(),
                             'recall':recall_score(grp.actual,grp.predicted,zero_division=0),
                             'precision':precision_score(grp.actual,grp.predicted,zero_division=0)}),
        include_groups=False).reset_index()
    by_channel.to_csv(reports/'stockout_channel_fairness.csv',index=False)
    cm=confusion_matrix(y,predictions,labels=[0,1]);fig_,ax=plt.subplots(figsize=(5,4))
    ax.imshow(cm,cmap='Blues')
    for i in range(2):
        for j in range(2):ax.text(j,i,f'{cm[i,j]:,}',ha='center',va='center')
    ax.set(xticks=[0,1],yticks=[0,1],xticklabels=['No stockout','Stockout'],
           yticklabels=['No stockout','Stockout'],xlabel='Predicted',ylabel='Actual',
           title='Held-out stockout prediction confusion matrix')
    fig_.tight_layout();fig_.savefig(fig/'stockout_confusion_matrix.png',dpi=155);plt.close(fig_)
    training_scaler=model.named_steps['preprocess'].named_transformers_['numeric'].named_steps['scaler']
    metadata={'train_period':'2021-01-01 through 2023-06-30 (after available lags)',
              'test_period':'2023-07-01 through 2023-12-31',
              'training_rows_used':int(len(training)),'test_rows':int(len(testing)),
              'categorical_features':CATEGORICAL,'numeric_features':NUMERIC,
              'scaler_fit':'training only','categorical_encoder_fit':'training only',
              'numeric_training_means':dict(zip(NUMERIC,map(float,training_scaler.mean_))),
              'average_precision':float(average_precision_score(y,probabilities)),
              'probability_threshold':0.5,
              'limitations':'Historical previous-observation lags can span several calendar days. '
                            'Recorded sales are subject to stockout censoring. '
                            'The evaluation uses actual test-period historical lag inputs (one-step-ahead). '
                            'No same-day stock_on_hand, units_sold, or lead_time_days is used as a predictor; '
                            'availability of calendar holiday flags at prediction time is assumed. '
                            'Class-balanced training and fixed threshold may produce many false alerts. '
                            'Stockout labels and sales are observational, not proof of causal stock savings.'}
    (reports/'preprocessing_and_stockout_notes.json').write_text(json.dumps(metadata,indent=2))
    print('GEOSPATIAL',len(locations),'stores, hotspots',int(locations.logistics_hotspot.sum()))
    print('CLASSIFICATION\n',metrics.round(4).to_string(index=False))
    print('CHANNELS\n',by_channel.round(4).to_string(index=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--zip',type=Path,default=ROOT.parent/'fmcg_raw_data.zip')
    main(p.parse_args().zip)
