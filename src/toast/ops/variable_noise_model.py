from astropy import units as u
import numpy as np
import traitlets

from .. import rng
from ..noise_sim import AnalyticNoise
from ..timing import function_timer
from ..traits import Bool, Float, Int, Unicode
from ..utils import Logger
from .operator import Operator


class VariableNoiseModel(Operator):
    """Create a noise model that varies from detector to detector."""

    # Class traits
    API = Int(0, help="Internal interface version for this operator")
    noise_model = Unicode(
        "var_noise_model", help="The observation key for storing the noise model"
    )
    pairs = Bool(False, help="Process detectors by pairs instead of individually")
    scatter = Float(0.1, help="Fractional scatter in the noise parameters")
    realization = Int(0, help="The model realization index")
    use_white = Bool(False, help="Use white noise instead of 1/f")
    uniform = Bool(
        False, help="Do not vary the noise parameters from detector to detector"
    )

    @traitlets.validate("realization")
    def _check_realization(self, proposal):
        check = proposal["value"]
        if check < 0:
            raise traitlets.TraitError("realization index must be positive")
        return check

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @function_timer
    def _exec(self, data, detectors=None, **kwargs):
        log = Logger.get()

        noise_keys = set(["psd_fmin", "psd_fknee", "psd_alpha", "psd_net"])

        for ob in data.obs:
            sindx = ob.session.uid
            telescope = ob.telescope.uid
            fp_data = ob.telescope.focalplane.detector_data
            has_parameters = False
            for key in noise_keys:
                if key not in fp_data.colnames:
                    break
            else:
                has_parameters = True
            if not has_parameters:
                msg = f"Observation {ob.name} does not have a focalplane with "
                msg += "noise parameters.  Skipping."
                log.warning(msg)
                ob[self.noise_model] = None
                continue

            local_dets = set(ob.local_detectors)

            dets = []
            fmin = {}
            fknee = {}
            alpha = {}
            NET = {}
            rates = {}
            indices = {}

            key1 = (
                int(self.realization) * int(4294967296)
                + int(telescope) * int(65536)
                + int(sindx)
            )

            def _process_row(row, second=False):
                name = row["name"]
                detindx = row["uid"]
                if name not in local_dets:
                    return
                dets.append(name)
                rates[name] = ob.telescope.focalplane.sample_rate
                if self.uniform:
                    coeff = np.ones(3) / np.sqrt(2)
                    if second:
                        coeff = -coeff
                else:
                    coeff = rng.random(3, sampler="gaussian", key=(key1, detindx))
                fmin[name] = row["psd_fmin"]
                if self.use_white:
                    fknee[name] = u.Quantity(0.0, u.Hz)
                    alpha[name] = 0.0
                else:
                    fknee[name] = row["psd_fknee"] * (1 + self.scatter * coeff[0])
                    alpha[name] = row["psd_alpha"] * (1 + self.scatter * coeff[1])
                NET[name] = row["psd_net"] * (1 + self.scatter * coeff[2])
                indices[name] = detindx

            if self.pairs:
                # iterate over pairs of detectors
                for row1, row2 in pairwise(fp_data):
                    _process_row(row1)
                    _process_row(row2, second=True)
            else:
                # iterate over single detectors
                for row in fp_data:
                    _process_row(row)

            ob[self.noise_model] = AnalyticNoise(
                rate=rates,
                fmin=fmin,
                detectors=dets,
                fknee=fknee,
                alpha=alpha,
                NET=NET,
                indices=indices,
            )

    def _finalize(self, data, **kwargs):
        return

    def _requires(self):
        return dict()

    def _provides(self):
        prov = {"meta": [self.noise_model]}
        return prov


def pairwise(iterable):
    """Iterate over pairs of elements in an iterable."""
    a = iter(iterable)
    return zip(a, a)
