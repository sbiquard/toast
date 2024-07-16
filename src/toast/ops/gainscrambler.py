# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import re

import traitlets

from .. import rng
from ..observation import default_values as defaults
from ..timing import function_timer
from ..traits import Bool, Float, Int, Unicode, List, trait_docs
from ..utils import Logger
from .operator import Operator


@trait_docs
class GainScrambler(Operator):
    """Apply random gain errors to detector data.

    This operator draws random gain errors from a given distribution and
    applies them to the specified detectors.
    """

    # Class traits

    API = Int(0, help="Internal interface version for this operator")

    det_data_names = List(
        trait=Unicode,
        default_value=[defaults.det_data],
        help="Observation detdata key(s) to apply the gain error to",
    )

    pattern = Unicode(
        f".*",
        allow_none=True,
        help="Regex pattern to match against detector names. Only detectors that "
        "match the pattern are scrambled.",
    )

    dist = Unicode("gaussian", allow_none=False, help="Gain distribution density")

    location = Float(1, allow_none=False, help="Distribution location parameter")

    scale = Float(1e-3, allow_none=False, help="Distribution scale parameter")

    realization = Int(0, allow_none=False, help="Realization index")

    component = Int(0, allow_none=False, help="Component index for this simulation")

    store = Bool(False, allow_none=False, help="Store the scrambled values")

    process_pairs = Bool(False, allow_none=False, help="Process detectors in pairs")

    constant = Bool(
        False,
        allow_none=False,
        help="If True, scramble all detector pairs in the same way",
    )

    @traitlets.validate("det_mask")
    def _check_dist(self, proposal):
        check = proposal["value"]
        valid = ["gaussian", "cauchy"]
        if check not in valid:
            raise traitlets.TraitError(
                "Invalid choice for trait 'dist' (must be one of {valid})"
            )
        return check

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @function_timer
    def _exec(self, data, detectors=None, **kwargs):
        log = Logger.get()

        if self.pattern is None:
            pat = None
        else:
            pat = re.compile(self.pattern)

        for obs in data.obs:
            # Get the detectors we are using for this observation
            dets = obs.select_local_detectors(detectors)
            if len(dets) == 0:
                # Nothing to do for this observation
                continue

            comm = obs.comm.comm_group
            rank = obs.comm.group_rank

            sindx = obs.session.uid
            telescope = obs.telescope.uid

            focalplane = obs.telescope.focalplane

            # key1 = realization * 2^32 + telescope * 2^16 + component
            key1 = self.realization * 4294967296 + telescope * 65536 + self.component
            key2 = sindx
            counter1 = 0
            counter2 = 0

            dets_present = {
                name: set(obs.detdata[name].detectors) for name in self.det_data_names
            }

            if self.store:
                obs.scrambled_gains = {}

            # Process by pairs
            if self.process_pairs:
                for det_a, det_b in pairwise(dets):
                    # Warn if the detectors don't look like a pair
                    if not det_b.startswith(det_a.removesuffix("A")):
                        log.warning_rank(
                            f"Detectors ({det_a=}, {det_b=}) don't look like a pair"
                        )

                    # Test the detector pattern
                    if pat is not None and (
                        pat.match(det_a) is None or pat.match(det_b) is None
                    ):
                        continue

                    detindx = focalplane[det_a]["uid"]
                    counter1 = detindx

                    if self.constant:
                        sample = 1.0
                    else:
                        sample = self._random_sample(key1, key2, counter1, counter2)

                    # Apply symmetric gains to detectors A and B
                    gain_a = self.loc + 0.5 * sample * self.scale
                    gain_b = self.loc - 0.5 * sample * self.scale

                    for name, det_set in dets_present.items():
                        if not set((det_a, det_b)).issubset(det_set):
                            continue

                        obs.detdata[name][det_a] *= gain_a
                        obs.detdata[name][det_b] *= gain_b

                        if self.store:
                            obs.scrambled_gains.update({det_a: gain_a, det_b: gain_b})

                continue

            # Standard processing
            for det in dets:
                # Test the detector pattern
                if pat is not None and pat.match(det) is None:
                    continue

                detindx = focalplane[det]["uid"]
                counter1 = detindx

                sample = self._random_sample(key1, key2, counter1, counter2)
                gain = self.loc + sample * self.scale

                for name, det_set in dets_present.items():
                    if not det in det_set:
                        continue

                    obs.detdata[name][det] *= gain

                    if self.store:
                        # save the applied gains
                        obs.scrambled_gains[det] = gain

        return

    def _random_sample(self, key1, key2, counter1, counter2):
        kc = {"key": (key1, key2), "counter": (counter1, counter2)}
        if self.dist == "gaussian":
            rngdata = rng.random(1, sampler="gaussian", **kc)
            return rngdata[0]
        if self.dist == "cauchy":
            from numpy import tan, pi

            rngdata = rng.random(1, sampler="uniform01", **kc)
            cauchy = tan(pi * (rngdata - 0.5))
            return cauchy[0]

    def _finalize(self, data, **kwargs):
        return

    def _requires(self):
        req = {
            "meta": list(),
            "shared": list(),
            "detdata": self.det_data_names,
            "intervals": list(),
        }
        return req

    def _provides(self):
        prov = {
            "meta": list(),
            "shared": list(),
            "detdata": list(),
        }
        return prov


def pairwise(iterable):
    """s -> (s0,s1), (s2,s3), (s4, s5), ..."""
    a = iter(iterable)
    return zip(a, a)
