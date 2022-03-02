import datetime
import lfi.backtesting.backtesting as backtesting

class BacktestEnv:
    def __init__(self, prepare_bars_fn, starting_cash=10000, comission=0.002, exclusive_orders=True, action_threshold=0.5, desired_gain=0.3, trade_market_hours_only=True):
        self.comission = comission
        self.starting_cash = starting_cash
        self.exclusive_orders = exclusive_orders
        self.desired_gain = desired_gain
        self.prepare_bars = prepare_bars_fn
        self.action_threshold = action_threshold
        self._observation = None
        self.allow_off_hours = not trade_market_hours_only

    def reset(self):
        self.simulation = backtesting.Backtest(self.prepare_bars(), BacktestEnv.Strategy,
                                    cash=self.starting_cash, commission=self.comission,
                                    exclusive_orders=self.exclusive_orders)
        self.simulation.reset()
        self.simulation.step()

        # TODO: NOTE: This WILL be a problem later since it's a list of samples and hence out-of-sample from the observation space (at least w.r.t dimensions)
        warmup = self.simulation.sim_data.df
        return [warmup.iloc[t] for t in range(len(warmup))]

    def step(self, action):
        '''NOTE: time between bars is supposdly large enough for a trade order to execute'''
        self._act(action)

        # Tick simulation
        sim_ended = self.simulation.step()
        
        # Inspect environment and quantify
        reward = self.strategy.step_reward
        if sim_ended and self.desired_gain is not None:
            reward *= 1 + ((self.cash - self.starting_cash) / self.starting_cash - self.desired_gain)

        observation = self.observation
        extras = {}

        return observation, reward, sim_ended, extras

    def _act(self, action):
        if not self.allow_off_hours and not self.markethours: return
        if action < -self.action_threshold and self.holds_position:
            self.position.close()
        elif action > self.action_threshold and not self.holds_position:
            nshares = self.cash // self.last_close
            if nshares > 0: self.strategy.buy(size=nshares)
    
    @property
    def now(self): return self.observation.name
    @property
    def last_close(self): return self.strategy.observation.Close
    @property
    def observation(self): return self.strategy.observation # TODO: Add observation!
    @property
    def holds_position(self): return self.strategy.position.size > 0
    @property
    def position(self): return self.strategy.position
    @property
    def cash(self): return self.broker._cash
    @property
    def bars(self): return self.strategy.data.df
    @property
    def strategy(self): return self.simulation.sim_strategy
    @property
    def broker(self): return self.simulation.sim_broker
    @property
    def premarket(self): return self.now.time() < datetime.time(hour=9, minute=30)
    @property
    def aftermarket(self): return self.now.time() > datetime.time(hour=16)
    @property
    def markethours(self): return not (self.premarket or self.aftermarket)

    class Strategy(backtesting.Strategy):
        ''' This class does nothing except bookeeping to keep the environment step clean(er),
            since the next() method is the last thing done by the Backtest step,
            the trading biz logic is part of the environment'''
        def init(self):
            self.last_trade_count = 0

        def next(self):
            self.observation = self.data.df.iloc[-1].copy()
            self.observation.timestamp = self.observation.timestamp.timestamp()
            self.step_reward = self.trades[-1].pl_pct if len(self.trades) > self.last_trade_count else 0
            self.last_trade_count = len(self.trades)

        