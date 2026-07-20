"""Twitch/force model invariants (Fuglevand)."""
import numpy as np
import pytest

from emgforge.activation import MotoneuronPool, TwitchPool, drive


def test_twitch_peaks_at_T_with_height_P():
    p = MotoneuronPool(n_mu=30, fs=2048)
    tw = TwitchPool(p, rp=100)
    for m in (0, 15, 29):
        peak_idx = int(np.argmax(tw.tw[m]))
        assert peak_idx == pytest.approx(tw.T[m], abs=1.5)          # peaks at contraction time
        assert tw.tw[m].max() == pytest.approx(tw.P[m], rel=0.02)   # to height P


def test_larger_units_are_bigger_and_faster():
    p = MotoneuronPool(n_mu=50)
    tw = TwitchPool(p, rp=100)
    assert tw.P[-1] > tw.P[0]          # larger unit → larger twitch
    assert tw.T[-1] < tw.T[0]          # …and faster (shorter contraction time)


def test_force_increases_with_drive_and_mvc_is_one():
    fs = 2048
    p = MotoneuronPool(n_mu=80, fs=fs)
    tw = TwitchPool(p, fs=fs)
    lv = [0.15, 0.4, 0.8]
    f = []
    for x in lv:
        E = drive.constant(x, 3.0, fs)
        f.append(tw.force(p.spike_trains(E, seed=1), n_samples=len(E))[fs:].mean())
    assert f[0] < f[1] < f[2]                                       # monotone with drive
    Efull = drive.constant(1.0, 4.0, fs)
    fmvc = tw.force(p.spike_trains(Efull, seed=0), n_samples=len(Efull))[fs:].mean()
    assert fmvc == pytest.approx(1.0, abs=0.1)                      # 100% drive ≈ 1 MVC


def test_fusion_gain_rises_with_rate():
    p = MotoneuronPool(n_mu=10)
    tw = TwitchPool(p)
    assert tw._gain(np.array([0.1]))[0] == pytest.approx(1.0, abs=1e-6)   # isolated
    assert tw._gain(np.array([1.5]))[0] > tw._gain(np.array([0.5]))[0]    # more fused
