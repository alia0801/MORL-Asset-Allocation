from gymnasium.envs.registration import register


# register(id="stock-portfolio-v0", entry_point="mo_gymnasium.envs.water_reservoir.dam_env:DamEnv")
register(id="stock-portfolio-v0", entry_point="stock_env.stock_env:StockPortfolioEnv")