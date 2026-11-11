"""Additional diagnostics: promotional association, price elasticity, equity and explainability."""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
from sklearn.ensemble import HistGradientBoostingRegressor
import statsmodels.formula.api as smf
from pipeline import ROOT, load_sales, build_daily, FEATURES

def run(zip_path):
    reports=ROOT/'reports'; figures=reports/'figures'
    sales=load_sales(zip_path)
    # Equity diagnostic by sales channel: measured stockout and service outcomes.
    equity=sales.groupby('channel').agg(records=('sku_id','size'),stockout_rate=('stock_out_flag','mean'),units=('units_sold','sum'),
        lead_time=('lead_time_days','mean'),promo_share=('promo_flag','mean')).reset_index()
    equity.to_csv(reports/'channel_equity.csv',index=False)
    axes=equity.set_index('channel').stockout_rate.mul(100).sort_values().plot.barh(figsize=(7,4))
    axes.set(xlabel='Records with stockout (%)',title='Stockout incidence by sales channel')
    plt.tight_layout();plt.savefig(figures/'channel_stockout.png',dpi=150);plt.close()
    sales['effective_price']=sales.list_price*(1-sales.discount_pct)
    sales['log_units']=np.log1p(sales.units_sold)
    sales['log_price']=np.log(sales.effective_price.clip(lower=.01))
    # Fixed effects across SKU and month, controlling promo. Association, not causal elasticity.
    sample=sales.sample(n=min(120000,len(sales)),random_state=42)
    fitted=smf.ols('log_units ~ log_price + promo_flag + C(sku_id) + C(month) + C(weekday)',data=sample).fit(cov_type='HC3')
    effect={'log_price_coefficient':float(fitted.params['log_price']), 'log_price_ci95':list(map(float,fitted.conf_int().loc['log_price'])),
            'p_value':float(fitted.pvalues['log_price']), 'sample_size':len(sample),
            'method':'OLS, log(1+units) ~ log(discount-adjusted price) + promotion + SKU/month/weekday fixed effects; HC3 robust SE',
            'caution':'Associational approximate elasticity only; store-level variation, endogeneity and stockouts not fully controlled.'}
    (reports/'price_association.json').write_text(json.dumps(effect,indent=2))
    del sales,sample
    chosen=pd.read_csv(reports/'model_metrics.csv').sku_id.drop_duplicates().tolist()
    daily=build_daily(load_sales(zip_path))
    daily=daily[daily.sku_id.eq(chosen[0])].dropna(subset=FEATURES+['units_sold'])
    tr=daily[daily.date<'2023-07-01'];te=daily[daily.date>='2023-07-01']
    model=HistGradientBoostingRegressor(max_iter=90,max_leaf_nodes=15,l2_regularization=2,random_state=42)
    model.fit(tr[FEATURES],tr.units_sold)
    pi=permutation_importance(model,te[FEATURES],te.units_sold,n_repeats=5,random_state=42,scoring='neg_mean_absolute_error')
    importance=pd.DataFrame({'feature':FEATURES,'mae_increase':pi.importances_mean}).sort_values('mae_increase',ascending=False)
    importance.to_csv(reports/'feature_importance_example.csv',index=False)
    ax=importance.set_index('feature').mae_increase.sort_values().plot.barh(figsize=(8,5));ax.set(xlabel='Increase in MAE when feature shuffled (units)',title=f'Permutation feature importance: {chosen[0]}')
    plt.tight_layout();plt.savefig(figures/'feature_importance.png',dpi=150);plt.close()
    print('PRICE',json.dumps(effect));print('CHANNELS\n',equity.to_string(index=False));print('FEATURE TOP\n',importance.head(5).to_string(index=False))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--zip',type=Path,default=ROOT.parent/'fmcg_raw_data.zip');run(parser.parse_args().zip)
