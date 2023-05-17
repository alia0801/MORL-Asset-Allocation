# %%
from contextlib import closing
from io import StringIO
from os import path
from typing import Optional

import gymnasium as gym
import numpy as np
import pygame
from gymnasium.spaces.box import Box
from gymnasium.utils import EzPickle
import yfinance as yf
from finrl.finrl_meta.finrl_meta_config import DOW_30_TICKER
from finrl.finrl_meta.data_processors.processor_yahoofinance import YahooFinanceProcessor
from finrl.apps.config import *
import pandas as pd
from config import *
from morl_baselines.common.evaluation import eval_mo_1time_info
import datetime
# %%

def get_df(asset_list):
    yfp = YahooFinanceProcessor()
    download_df = yfp.download_data(start_date = TRAIN_START_DATE,
                         end_date = TEST_END_DATE,
                         ticker_list = asset_list,
                         time_interval='1D')
    # df = yf.download(DOW_30_TICKER,start=TRAIN_START_DATE,end=TEST_END_DATE)
    df = yfp.clean_data(download_df)

    # df.rename(columns = {'date':'time'}, inplace = True)
    df = yfp.add_technical_indicator(df, TECHNICAL_INDICATORS_LIST)

    # df.rename(columns = {'time':'date'}, inplace = True)
    df['date'] = df['time']
    df = yfp.add_turbulence(df)

    df = yfp.add_vix(df)


    df=df.sort_values(['date','tic'],ignore_index=True)
    df.index = df.date.factorize()[0]

    cov_list = []
    return_list = []

    # look back is one year
    lookback=252
    for i in range(lookback,len(df.index.unique())):
      data_lookback = df.loc[i-lookback:i,:]
      price_lookback=data_lookback.pivot_table(index = 'date',columns = 'tic', values = 'close')
      return_lookback = price_lookback.pct_change().dropna()
      return_list.append(return_lookback)

      covs = return_lookback.cov().values 
      cov_list.append(covs)
    
    df_cov = pd.DataFrame({'date':df.date.unique()[lookback:],'cov_list':cov_list,'return_list':return_list})
    df = df.merge(df_cov, on='date')
    df = df.sort_values(['date','tic']).reset_index(drop=True)
    return df

def evaluate_agent(agent,env,w,time_df,mode = 'train'):
    
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H_%M_%S")
    scalarized_return,scalarized_discounted_return,vec_return,disc_vec_return,info = eval_mo_1time_info(agent,env,w)
    ann_reward,sharpe,asset_memory = info
    asset_memory_df = pd.DataFrame(asset_memory,columns=['money'])
    res = pd.concat([time_df,asset_memory_df],axis=1, ignore_index=True)
    res.columns=['date','money']
    res.to_csv('./asset_memory/asset_memory_df_'+mode+timestamp+'.csv')
    print(mode)
    print('ann_reward',ann_reward)
    print('sharpe',sharpe)
    return ann_reward, sharpe

