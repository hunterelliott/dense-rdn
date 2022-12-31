"""
Test's of DriveLayer classes
"""
import pytest
import numpy as np

from models import drive


@pytest.fixture(params=[[1], [0, -1]])
def drive_chans(request):
    return np.array(request.param, dtype=np.int32)


def test_flow_drive(n_chan_numpy_x_0, drive_chans):
    """
    Just test that the output is the right shape and indexing
    """

    n_chan_numpy_x_0 = (n_chan_numpy_x_0 + 1) / 2
    feed_conc = np.zeros(n_chan_numpy_x_0.shape[-1])
    feed_conc[drive_chans] = np.random.uniform(.6, .9, len(drive_chans))
    flow_rate = .234

    drive_layer = drive.FlowDriveLayer(init_feed_conc=feed_conc, init_flow_rate=flow_rate)

    drive_dx_dt = drive_layer(n_chan_numpy_x_0).numpy()

    assert drive_dx_dt.shape == n_chan_numpy_x_0.shape
    # Make sure we drove the right channels
    assert np.allclose(drive_dx_dt[:, :, :, drive_chans], flow_rate * (np.reshape(feed_conc[drive_chans], (1, 1, 1, -1)) - n_chan_numpy_x_0[:, :, :, drive_chans]))
    no_drive = np.delete(drive_dx_dt, drive_chans, axis=3)
    assert np.all(no_drive == np.delete(n_chan_numpy_x_0, drive_chans, axis=3)*-flow_rate)


def test_noisy_flow_drive(n_chan_numpy_x_0, drive_chans):
    """
    Just test that the output is the right shape and indexing
    """

    n_chan_numpy_x_0 = (n_chan_numpy_x_0 + 1) / 2
    feed_conc = np.zeros(n_chan_numpy_x_0.shape[-1])
    feed_conc[drive_chans] = np.random.uniform(.6, .9, len(drive_chans))
    flow_rate = .234
    noise_amp = flow_rate / 2

    drive_layer = drive.NoisyFlowDriveLayer(init_feed_conc=feed_conc, init_flow_rate=flow_rate,
                                            noise_amplitude=noise_amp)

    drive_dx_dt = drive_layer(n_chan_numpy_x_0).numpy()

    assert drive_dx_dt.shape == n_chan_numpy_x_0.shape
    # Even with the noise we should get only negative drive on un-driven channels
    no_drive = np.delete(drive_dx_dt, drive_chans, axis=3)
    assert np.all(no_drive <= 0)

    # Make sure we're getting different values each call
    drive_dx_dt_2 = drive_layer(n_chan_numpy_x_0).numpy()
    assert not np.allclose(drive_dx_dt_2, drive_dx_dt)

    # And that the randomness is as expected - varies per sample and per channel
    drive_layer = drive.NoisyFlowDriveLayer(init_feed_conc=np.ones_like(feed_conc), init_flow_rate=flow_rate,
                                            noise_amplitude=noise_amp)

    x_0_same = np.stack([n_chan_numpy_x_0[:, :, :, 0] for _ in feed_conc], axis=-1)
    x_0_same = np.stack([x_0_same[0] for _ in range(n_chan_numpy_x_0.shape[0])], axis=0)

    drive_dx_dt = drive_layer(x_0_same).numpy()
    assert not np.allclose(drive_dx_dt[0], drive_dx_dt[1])
    assert not np.allclose(drive_dx_dt[0, :, :, 0], drive_dx_dt[0, :, :, 1])

    # Make the amplitude too large and make sure we don't get negative flow
    noise_amp = 1e4*flow_rate
    drive_layer = drive.NoisyFlowDriveLayer(init_feed_conc=feed_conc, init_flow_rate=flow_rate,
                                            noise_amplitude=noise_amp)
    drive_dx_dt = drive_layer(n_chan_numpy_x_0).numpy()
    no_drive = np.delete(drive_dx_dt, drive_chans, axis=3)
    assert np.all(no_drive <= 0)


def test_synthesis_drive(n_chan_numpy_x_0, drive_chans):
    """
    Just test that the output is the right shape and indexing
    """

    n_chan_numpy_x_0 = (n_chan_numpy_x_0 + 1) / 2
    synth_rate = np.zeros(n_chan_numpy_x_0.shape[-1])
    synth_rate[drive_chans] = np.random.uniform(.6, .9, len(drive_chans))

    drive_layer = drive.ConstantSynthesisDriveLayer(init_synth_rate=synth_rate)

    drive_dx_dt = drive_layer(n_chan_numpy_x_0).numpy()

    assert len(drive_dx_dt.shape) == len(n_chan_numpy_x_0.shape)
    assert drive_dx_dt.shape[-1] == n_chan_numpy_x_0.shape[-1]
    assert drive_dx_dt.size == n_chan_numpy_x_0.shape[-1]
    # Test the broadcasting addition
    x_t_plus_1 = n_chan_numpy_x_0 + drive_dx_dt

    # Make sure we drove the right channels. I know, I'm paranoid
    assert np.allclose(drive_dx_dt, synth_rate)
    assert np.all(np.delete(drive_dx_dt, drive_chans, axis=3) == 0)