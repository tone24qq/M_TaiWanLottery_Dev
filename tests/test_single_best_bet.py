import numpy as np
from scripts.bingo_single_best_bet_backtest import features,interval

def test_features_exclude_target_and_future():
 x=np.random.default_rng(3).integers(0,2,(100,80)).astype(float)
 before=features(x,70,20);x[70:]=1-x[70:]
 np.testing.assert_array_equal(before,features(x,70,20))

def test_wilson_reference():
 lo,hi=interval(250,1000)
 assert lo<.25<hi and .22<lo<.23 and .27<hi<.28

def test_neighbor_boundaries_do_not_wrap():
 x=np.zeros((30,80));x[:,79]=1
 f=features(x,25,20)
 assert f[0,10]==0 and f[78,10]==.5
