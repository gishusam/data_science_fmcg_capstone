"""Reproducible analysis of the supplied FMCG transaction CSV inside its ZIP archive."""
from __future__ import annotations
import argparse, json, zipfile, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.stattools import adfuller

ROOT=Path(__file__).resolve().parents[1]
FEATURES=['lag_1','lag_7','lag_30','rolling_7','day_of_week','month','is_weekend','is_holiday','is_month_start','is_month_end','promo_rate','effective_price','rain_mm','temperature']

def load_sales(path:Path):
    with zipfile.ZipFile(path) as z:
        with z.open('data/fmcg_sales_3years_1M_rows.csv') as f:
            return pd.read_csv(f,parse_dates=['date'])

def build_daily(df):
    daily=df.groupby(['sku_id','date'], observed=True).agg(
        units_sold=('units_sold','sum'), net_sales=('net_sales','sum'),
        promo_rate=('promo_flag','mean'), effective_price=('list_price', 'mean'),
        avg_discount=('discount_pct','mean'), rain_mm=('rain_mm','mean'),
        temperature=('temperature','mean'), is_holiday=('is_holiday','max'),
        stockout_rate=('stock_out_flag','mean')).reset_index().sort_values(['sku_id','date'])
    # Mean transacted price including discount. Price is assumed observed, not a future-known feature.
    daily['effective_price']=daily.effective_price*(1-daily.avg_discount)
    daily['day_of_week']=daily.date.dt.dayofweek
    daily['month']=daily.date.dt.month
    daily['is_weekend']=(daily.day_of_week>=5).astype(int)
    daily['is_month_start']=daily.date.dt.is_month_start.astype(int)
    daily['is_month_end']=daily.date.dt.is_month_end.astype(int)
    for lag in (1,7,30):
        daily[f'lag_{lag}']=daily.groupby('sku_id').units_sold.shift(lag)
    daily['rolling_7']=daily.groupby('sku_id').units_sold.transform(lambda x:x.shift(1).rolling(7,min_periods=7).mean())
    return daily

def save_plot_daily(df,out):
    ax=df.groupby('date').units_sold.sum().plot(figsize=(11,4),lw=1.3)
    ax.set(title='Daily units sold across available stores (2021–2023)',ylabel='Units',xlabel='Date')
    plt.tight_layout();plt.savefig(out,dpi=150);plt.close()

def main(zip_path):
    reports=ROOT/'reports';fig=reports/'figures';fig.mkdir(parents=True,exist_ok=True)
    df=load_sales(zip_path)
    audit={'rows':int(len(df)),'columns':int(df.shape[1]),'date_start':str(df.date.min().date()),'date_end':str(df.date.max().date()),
        'countries':int(df.country.nunique()),'cities':int(df.city.nunique()),'stores':int(df.store_id.nunique()),
        'skus':int(df.sku_id.nunique()),'channels':df.channel.value_counts().to_dict(),
        'missing_cells':int(df.isna().sum().sum()),'full_duplicates':int(df.duplicated().sum()),
        'duplicate_store_sku_day':int(df.duplicated(['date','store_id','sku_id']).sum()),
        'duplicate_city_channel_sku_day':int(df.duplicated(['date','city','channel','sku_id']).sum()),
        'stockout_rate':float(df.stock_out_flag.mean()),'promotion_rate':float(df.promo_flag.mean())}
    (reports/'data_audit.json').write_text(json.dumps(audit,indent=2))
    save_plot_daily(df,fig/'daily_sales.png')
    # Pareto revenue ranking: tie-aware boundary on cumulative contributions.
    sku=df.groupby('sku_id').agg(revenue=('net_sales','sum'),units=('units_sold','sum')).sort_values('revenue',ascending=False)
    sku['revenue_share']=sku.revenue/sku.revenue.sum()
    sku['cumulative_share']=sku.revenue_share.cumsum()
    sku['previous_share']=sku.cumulative_share-sku.revenue_share
    sku['ABC']=np.select([sku.previous_share<0.8,sku.previous_share<0.95],['A','B'],default='C')
    # Weekly demand CV, excluding incomplete missing weeks only if there is no recorded demand;
    # treated as recorded sales not unconstrained latent demand.
    weekly=df.assign(week=df.date.dt.to_period('W').dt.start_time).groupby(['sku_id','week']).units_sold.sum()
    cv=weekly.groupby(level=0).agg(['mean','std']);cv['cv']=(cv['std']/cv['mean'].replace(0,np.nan))
    sku=sku.join(cv['cv'])
    # Explicit thresholds are heuristic assumptions, not mandated by the coursework.
    sku['XYZ']=np.select([sku.cv<=0.5,sku.cv<=1.0],['X','Y'],default='Z')
    sku['segment']=sku.ABC+sku.XYZ
    sku.reset_index().to_csv(reports/'sku_segmentation.csv',index=False)
    counts=sku.groupby(['ABC','XYZ']).size().unstack(fill_value=0).reindex(index=['A','B','C'],columns=['X','Y','Z'],fill_value=0)
    fig_,ax=plt.subplots(figsize=(6,4));im=ax.imshow(counts.values,cmap='Blues')
    ax.set_xticks(range(3),['X','Y','Z']);ax.set_yticks(range(3),['A','B','C']);ax.set(xlabel='Demand variability',ylabel='Revenue class',title='SKU counts by ABC/XYZ segment')
    for i in range(3):
        for j in range(3): ax.text(j,i,int(counts.iloc[i,j]),ha='center',va='center')
    fig_.tight_layout();fig_.savefig(fig/'abc_xyz.png',dpi=150);plt.close(fig_)
    region=df.groupby('country').agg(units=('units_sold','sum'),revenue=('net_sales','sum'),stockout_rate=('stock_out_flag','mean'),mean_lead_time=('lead_time_days','mean')).sort_values('revenue',ascending=False)
    region.to_csv(reports/'country_metrics.csv')
    ax=region.revenue.sort_values().plot.barh(figsize=(8,4));ax.set(xlabel='Net sales (currency unspecified)',title='Net sales by country');plt.tight_layout();plt.savefig(fig/'country_revenue.png',dpi=150);plt.close()
    # Promotion comparison is observational, not a causal A/B test.
    promoted=df.loc[df.promo_flag.eq(1),'units_sold'];normal=df.loc[df.promo_flag.eq(0),'units_sold']
    welch=stats.ttest_ind(promoted,normal,equal_var=False)
    promo={'mean_units_promo':float(promoted.mean()),'mean_units_nonpromo':float(normal.mean()),
           'relative_difference_pct':float((promoted.mean()/normal.mean()-1)*100),
           'welch_t_statistic':float(welch.statistic),'p_value':float(welch.pvalue),
           'interpretation':'Observational association only; promotions may be targeted to high-demand stores/dates.'}
    (reports/'promotion_test.json').write_text(json.dumps(promo,indent=2))
    # Regional logistics risk: grouped by store, not claims of causality.
    logistics=df.groupby(['country','city','store_id']).agg(latitude=('latitude','first'),longitude=('longitude','first'),
        mean_lead=('lead_time_days','mean'),lead_sd=('lead_time_days','std'),stockout_rate=('stock_out_flag','mean'),units=('units_sold','sum')).reset_index()
    logistics.to_csv(reports/'logistics_store_metrics.csv',index=False)
    fig_,ax=plt.subplots(figsize=(7,4));ax.scatter(logistics.lead_sd,logistics.stockout_rate,s=np.clip(logistics.units/2000,20,170),alpha=.6)
    ax.set(xlabel='Within-store lead-time SD (days)',ylabel='Stockout rate',title='Lead-time variability and stockout rate by store')
    fig_.tight_layout();fig_.savefig(fig/'logistics_risk.png',dpi=150);plt.close(fig_)
    del df
    daily=build_daily(load_sales(zip_path))
    # Choose diverse feasible groups, with fallback to remaining top-revenue SKUs.
    chosen=[]
    for seg in ['AX','BY','CZ','AY','BX','CY','AZ','BZ','CX']:
        options=sku[sku.segment.eq(seg)].index.tolist()
        chosen.extend([s for s in options if s not in chosen][:2 if seg in ['AX','BY','CZ'] else 1])
        if len(chosen)>=10:break
    if len(chosen)<10:chosen += [s for s in sku.index if s not in chosen][:10-len(chosen)]
    chosen=chosen[:10]
    daily=daily[daily.sku_id.isin(chosen)].dropna(subset=FEATURES+['units_sold']).copy()
    # Last six calendar months are held out; forecasts are one-step ahead with actual lag values.
    test_start=pd.Timestamp('2023-07-01')
    train=daily[daily.date<test_start];test=daily[daily.date>=test_start]
    results=[];predictions=[];adf_results=[]
    for sid in chosen:
        tr=train[train.sku_id.eq(sid)];te=test[test.sku_id.eq(sid)]
        if len(tr)<90 or len(te)<30:continue
        try:
            adf=adfuller(tr.units_sold.values,autolag='AIC',maxlag=14)
            adf_results.append({'sku_id':sid,'adf_statistic':float(adf[0]),'p_value':float(adf[1]),'stationary_at_005':bool(adf[1]<.05)})
        except ValueError:adf_results.append({'sku_id':sid,'adf_statistic':None,'p_value':None,'stationary_at_005':None})
        models={'Linear Regression':Pipeline([('scaler',StandardScaler()),('regression',LinearRegression())]), 'Histogram Gradient Boosting':HistGradientBoostingRegressor(max_iter=90,max_leaf_nodes=15,l2_regularization=2,random_state=42)}
        for model_name,model in models.items():
            model.fit(tr[FEATURES],tr.units_sold)
            forecast=np.maximum(0,model.predict(te[FEATURES]))
            mae=mean_absolute_error(te.units_sold,forecast)
            rmse=np.sqrt(mean_squared_error(te.units_sold,forecast))
            # MAPE only on strictly positive actual demand; also report WAPE for robustness.
            valid=te.units_sold.to_numpy()>0
            mape=float(np.mean(np.abs((te.units_sold.to_numpy()[valid]-forecast[valid])/te.units_sold.to_numpy()[valid]))*100) if valid.any() else np.nan
            wape=float(np.abs(te.units_sold.to_numpy()-forecast).sum()/te.units_sold.sum()*100) if te.units_sold.sum()>0 else np.nan
            results.append({'sku_id':sid,'segment':sku.loc[sid,'segment'],'model':model_name,'train_rows':len(tr),'test_rows':len(te),
                            'MAE':mae,'RMSE':rmse,'MAPE_pct_nonzero':mape,'WAPE_pct':wape})
            predictions.append(pd.DataFrame({'sku_id':sid,'date':te.date.values,'actual':te.units_sold.values,'predicted':forecast,'model':model_name}))
    metrics=pd.DataFrame(results);metrics.to_csv(reports/'model_metrics.csv',index=False)
    pd.DataFrame(adf_results).to_csv(reports/'adf_stationarity.csv',index=False)
    pred=pd.concat(predictions,ignore_index=True);pred.to_csv(reports/'test_predictions.csv',index=False)
    summary=metrics.groupby('model')[['MAE','RMSE','MAPE_pct_nonzero','WAPE_pct']].mean()
    summary.to_csv(reports/'model_comparison.csv')
    ax=summary.MAE.sort_values().plot.bar(figsize=(7,4));ax.set(ylabel='Mean SKU-level MAE (units)',title='Model comparison: 10 selected SKUs');plt.xticks(rotation=15,ha='right');plt.tight_layout();plt.savefig(fig/'model_comparison.png',dpi=150);plt.close()
    first=chosen[0];sample=pred[(pred.sku_id==first)&(pred.model=='Histogram Gradient Boosting')].tail(70)
    fig_,ax=plt.subplots(figsize=(10,4));ax.plot(sample.date,sample.actual,label='Actual',lw=1.2);ax.plot(sample.date,sample.predicted,label='Predicted',lw=1.2)
    ax.set(title=f'{first}: final 70 test days (one-step ahead)',ylabel='Daily units sold');ax.legend();fig_.autofmt_xdate();fig_.tight_layout();fig_.savefig(fig/'forecast_example.png',dpi=150);plt.close(fig_)
    print('AUDIT',json.dumps(audit));print('SELECTED',chosen);print('MODEL SUMMARY\n',summary.round(2).to_string());print('PROMO',promo)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--zip',type=Path,default=ROOT.parent/'fmcg_raw_data.zip');args=ap.parse_args();main(args.zip)
