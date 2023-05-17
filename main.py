
import logging
from finrl.apps.config import *
from finrl.preprocessing.data import data_split
from gymnasium.envs.registration import register

from morl_baselines.multi_policy.pgmorl.pgmorl import PGMORL
from morl_baselines.multi_policy.gpi_pd.gpi_pd_continuous_action import GPIPDContinuousAction
from morl_baselines.multi_policy.linear_support.linear_support import LinearSupport
from config import *
from util import *


if __name__ == '__main__':

    log_file_timestr = datetime.datetime.now().strftime("%Y-%m-%d_%H_%M_%S")
    log_filename = log_file_timestr+'.log' #datetime.datetime.now().strftime("%Y-%m-%d_%H_%M_%S.log")
    logging.basicConfig(level=logging.INFO, filename='./log/'+log_filename, filemode='w',
	format='[%(asctime)s %(levelname)-8s] %(message)s',
	datefmt='%Y%m%d %H:%M:%S',
	)

    asset_list = DOW_30_TICKER[:4]
    weight_selection_algo='ols'
    num_eval_episodes_for_front=5

    logging.info('len_asset_list: %d'%(len(asset_list)))
    logging.info('asset_list: '+(' '.join(asset_list) ) )
    logging.info('TRAIN_START_DATE: '+TRAIN_START_DATE)
    logging.info('TRAIN_END_DATE: '+TRAIN_END_DATE)
    logging.info('TRAIN_START_DATE: '+TEST_START_DATE)
    logging.info('TRAIN_END_DATE: '+TEST_END_DATE)
    logging.info('len_TECHNICAL_INDICATORS_LIST: %d'% (len(TECHNICAL_INDICATORS_LIST) ))

    processed = get_df(asset_list)
    # print(processed)
    # print(processed.columns)
    

    stock_dimension = len(processed.tic.unique())
    # state_space = 1 + 2*stock_dimension + len(TECHNICAL_INDICATORS_LIST)*stock_dimension
    state_space = stock_dimension + len(TECHNICAL_INDICATORS_LIST)
    print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")
    logging.info(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")

    train_df = data_split(processed, TRAIN_START_DATE,TRAIN_END_DATE)
    test_df = data_split(processed, TEST_START_DATE,TEST_END_DATE)
    # print(train_df)
    # print(train_df.columns)

    train_time = pd.DataFrame(train_df.time.unique(),columns=['date'])
    test_time = pd.DataFrame(test_df.time.unique(),columns=['date'])
    # print(train_time)
    # print(test_time)

    env_kwargs = {
        "df":train_df,
        "hmax": 100, 
        "initial_amount": 1000000, 
        "transaction_cost_pct": 0.001, 
        # "sell_cost_pct": 0.001, 
        "state_space": state_space, 
        "stock_dim": stock_dimension, 
        "tech_indicator_list": TECHNICAL_INDICATORS_LIST,
        "action_space": stock_dimension, 
        "reward_scaling": 1e-4,
        # "print_verbosity":5
    }
    eval_env_kwargs = {
        "df":test_df,
        "hmax": 100, 
        "initial_amount": 1000000, 
        "transaction_cost_pct": 0.001, 
        # "sell_cost_pct": 0.001, 
        "state_space": state_space, 
        "stock_dim": stock_dimension, 
        "tech_indicator_list": TECHNICAL_INDICATORS_LIST,
        "action_space": stock_dimension, 
        "reward_scaling": 1e-4,
        # "print_verbosity":5
    }


    register(id="stock-portfolio-v0", entry_point="stock_env.stock_env:StockPortfolioEnv")
    env = gym.make("stock-portfolio-v0",**env_kwargs)
    eval_env = gym.make("stock-portfolio-v0",**eval_env_kwargs)

    agent = GPIPDContinuousAction(env=env)
    ref_point = np.array([-100.0, -100.0])

    agent.train(
            total_timesteps=10000,
            # log_every=100,
            # action_eval="hypervolume",
            # known_pareto_front=env.pareto_front(gamma=0.99),
            ref_point=ref_point,
            eval_env=env
        )
    linear_support = LinearSupport(num_objectives=reward_dim, epsilon=0.0 if weight_selection_algo == "ols" else None)
    obs, info = env.reset()
    w = linear_support.next_weight(
                        algo="gpi-ls", gpi_agent=agent, env=env, rep_eval=num_eval_episodes_for_front
                    )

    train_ann_reward, train_sharpe = evaluate_agent(agent,env,w, train_time,mode = 'train')
    logging.info('train_ann_reward: %.6f'%train_ann_reward)
    logging.info('train_sharpe: %.6f'%train_sharpe)

    test_ann_reward, test_sharpe = evaluate_agent(agent,eval_env,w, test_time, mode = 'test')
    logging.info('test_ann_reward: %.6f'%test_ann_reward)
    logging.info('test_sharpe: %.6f'%test_sharpe)