import kats
from kats.utils.decomposition import TimeSeriesDecomposition
from kats.consts import TimeSeriesData
from kats.detectors.cusum_detection import CUSUMDetector
from kats.detectors.bocpd import BOCPDetector, BOCPDModelType, TrendChangeParameters

# use pyenv environment disentangle_emotions
# scipy version = 1.7.3
# pandas version = 1.3.5


def changepoint_detection(df,time_colname,var_colnames,title=''):
    changepoints = {}
    for c in var_colnames:
        # construct ts
        tmp_ts = df.reset_index()
        tmp_ts = tmp_ts[[time_colname,c]]
        tmp_ts[time_colname] = tmp_ts[time_colname].astype(str)
        tmp_ts.columns = ['time','value']
        ts = TimeSeriesData(tmp_ts)
        
        cp_list = []

        detector = BOCPDetector(ts)
        # BOCPD - assume normal distri
        cp_list.extend(detector.detector(
            model=BOCPDModelType.NORMAL_KNOWN_MODEL,
            changepoint_prior = 0.03,
            threshold=0.6
        ))
#         # BOCPD - assume ordinary linear reg. 
#         cp_list.extend(detector.detector(
#             model=BOCPDModelType.TREND_CHANGE_MODEL,
#             model_parameters=TrendChangeParameters(
#                 readjust_sigma_prior=True, num_points_prior=14
#                 ),
#                 debug=True,
#                 threshold=0.6,
#                 choose_priors=False,
#                 agg_cp=True
#         ))
    
        # CUSUM - multiple change points
        historical_window = 14
        scan_window = 7
        step = 5
        cpts = []
        n = len(ts)
        for end_idx in range(historical_window + scan_window, n, step):
            tsd = ts[end_idx - (historical_window + scan_window) : end_idx]
            cpts += CUSUMDetector(tsd).detector(interest_window=[historical_window, historical_window + scan_window])
        
        # Plot the data, add results
        plt.figure(figsize=[20,3])
        plt.title(title+" - "+c)
        detector.plot(cp_list)
        plt.figure(figsize=[20,3])
        detector1 = CUSUMDetector(ts) # we are not really using this detector
        detector1.detector()
        detector1.plot(cpts)
        cp_list.extend(cpts)

        cleaned_list = []
        for j in cp_list:
            try:
                time = pd.Timestamp(j[0].start_time,tz='utc')
            except:
                time = pd.Timestamp(j[0].start_time)
            cleaned_list.append((time,j[0].confidence))
            print(time, j[0].confidence)
        
        idx_to_remove = []
        for j in range(1,len(cleaned_list)):
            if abs((cleaned_list[j][0] - cleaned_list[j-1][0]).days) <= 5:
                idx_to_remove.append(j)
        print(idx_to_remove)
        cleaned_list = [item for idx,item in enumerate(cleaned_list) if idx not in idx_to_remove]
        print(cleaned_list)
            
        changepoints[c] = cleaned_list
    return changepoints


def detect_event(time,events):
    for e in events:
        if time >= pd.Timestamp(e[0])-pd.Timedelta(1,unit='D') and time <= pd.Timestamp(e[0])+pd.Timedelta(1,unit='D'):
            event_date = e[0]
            return True,e
    return False,None


def measure_mean_change(df_ts,time_colname,var_colname,event_time,before_window=7,after_window=7):
    try:
        event_time = pd.Timestamp(event_time, tz='utc')
    except:
        event_time = pd.Timestamp(event_time)
    start_time = event_time - pd.Timedelta(before_window,unit='D')
    end_time = event_time + pd.Timedelta(after_window,unit='D')

    before_mean = df_ts.loc[(df_ts[time_colname]>=start_time) & (df_ts[time_colname]<event_time),var_colname].mean()
    after_mean = df_ts.loc[(df_ts[time_colname]>=event_time) & (df_ts[time_colname]<end_time),var_colname].mean()
    
    return (after_mean-before_mean)/before_mean*100


import statsmodels.formula.api as smf

def measure_rdd_change(df_ts,time_colname,var_colname,event_time,before_window=7,after_window=7,mode='kink',plot=True):
    if mode == 'jump':
        effect_coef = 'threshold'
    elif mode == 'kink':
        effect_coef = 'date_to_int:threshold'
    
    try:
        event_time = pd.Timestamp(event_time, tz='UTC')
    except:
        event_time = pd.Timestamp(event_time)
    start_time = event_time - pd.Timedelta(before_window,unit='D')
    end_time = event_time + pd.Timedelta(after_window,unit='D')
    
    df = df_ts[(df_ts[time_colname]>=start_time) & (df_ts[time_colname]<=end_time)]
    df = df.sort_index()
    df['date_to_int'] = list(range(len(df)))
    event_idx = df.loc[df[time_colname]==event_time,'date_to_int'].item()
    df['date_to_int'] = df['date_to_int'] - event_idx # make date_to_int of the event zero
    df = df.assign(threshold=(df['date_to_int'] > 0).astype(int))
    
    model = smf.wls("Q('"+var_colname+"')~date_to_int*threshold", df).fit()
    ate_pct = round(100*((model.params[effect_coef] + model.params["Intercept"])/model.params["Intercept"] - 1),2)

    # plot each regression - data and prediction
#     if plot:
#         plt.figure()
#         df_ = df.copy()
#         df_['predictions'] = model.predict()
#         df_.plot(x=time_colname, y="predictions", color="red")
#         df_.plot(kind='scatter',x=time_colname, y=var_colname)
#         plt.title(var_colname+f" ATE={ate_pct}%")

#     res.loc['effect',c] = model.params[effect_coef]
#     res.loc['p-val',c] = model.pvalues[effect_coef]
#     res.loc['std_err',c] = model.bse[effect_coef]
#     res.loc['change(%)',c] = ate_pct
    
    return ate_pct, model.pvalues[effect_coef]