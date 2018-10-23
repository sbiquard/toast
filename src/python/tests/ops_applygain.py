# Copyright (c) 2015-2017 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

from ..mpi import MPI
from .mpi import MPITestCase

import os

from ..dist import *
from ..tod.tod import *

from ..tod.applygain import *
from ._helpers import create_distdata, boresight_focalplane

class TestApplyGain(MPITestCase):

    def setUp(self):
        self.gain = {
                "TIME":np.arange(10),
                "1a":2*np.ones(10, dtype=np.float32),
                "1b":3*np.ones(10, dtype=np.float32)
                }
        self.gain["1b"][5:] = 0

    def test_write_calibration_file(self):
        # Only one process should write, otherwise they will
        # stomp on each other.
        if self.comm.rank == 0:
            write_calibration_file("test_cal.fits", self.gain)

    def test_op_applygain(self):

        # Note: self.comm is set by the test infrastructure

        self.toastcomm = Comm(world=self.comm)
        self.data = create_distdata(self.comm, obs_per_group=1)

        # Two detectors, default properties
        self.ndet = 2
        self.dets = ["1a", "1b"]
        dnames, dquat, depsilon, drate, dnet, dfmin, dfknee, dalpha = \
            boresight_focalplane(self.ndet)

        # Pick some number of samples per observation
        self.samples_per_obs = 10

        # (there is only one observation per group- see above)
        self.data.obs[0]['tod'] = TODCache(self.data.comm.comm_group,
            self.dets, self.samples_per_obs)

        # generate input data of ones for each of the 2 channels and 10 timestamps
        # from 1 to 10, no need for pointing

        for obs in self.data.obs:
            tod = obs['tod']
            # create timestamps for our local samples
            time_stamps = np.arange(tod.local_samples[1], dtype=np.float64)
            time_stamps += tod.local_samples[0]
            tod.write_times(stamps=time_stamps)
            # write our local data
            for det in tod.local_dets:
                tod.write(detector=det, data=np.ones(tod.local_samples[1]))


        op_apply_gain = OpApplyGain(self.gain, name="toast_tod_detdata")
        op_apply_gain.exec(self.data)

        # compare calibrated timelines

        for obs in self.data.obs:
            tod = obs['tod']
            locoff = tod.local_samples[0]
            nloc = tod.local_samples[1]
            # check our local pieces of data
            check = 2.0 * np.ones(nloc)
            np.testing.assert_allclose(check, tod.read(detector="1a"))
            check = np.zeros(self.samples_per_obs)
            check[0:5] = 3
            localcheck = check[locoff:locoff+nloc]
            np.testing.assert_allclose(localcheck, tod.read(detector="1b"))
