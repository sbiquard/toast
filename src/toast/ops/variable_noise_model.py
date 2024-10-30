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
    pairs = Bool(True, help="Process detectors by pairs instead of individually")
    scatter = Float(0.1, help="Fractional scatter in the noise parameters")
    realization = Int(0, help="The model realization index")
    use_white = Bool(False, help="Use white noise instead of 1/f")
    vary = Bool(True, help="Vary the noise parameters from detector to detector")

    # if `vary` is True then `pairs` does not matter
    # if `vary` is False and `pairs` is True then all pairs are the same (but not detectors inside pairs)

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

            def _process_row(row, key2=None):
                name = row["name"]
                detindx = row["uid"]
                if name not in local_dets:
                    return
                dets.append(name)
                rates[name] = ob.telescope.focalplane.sample_rate
                key2 = key2 if key2 is not None else detindx
                rngdata = rng.random(3, sampler="gaussian", key=(key1, key2))
                fmin[name] = row["psd_fmin"]
                if self.use_white:
                    fknee[name] = 0
                    alpha[name] = 0
                else:
                    fknee[name] = row["psd_fknee"] * (1 + self.scatter * rngdata[0])
                    alpha[name] = row["psd_alpha"] * (1 + self.scatter * rngdata[1])
                NET[name] = row["psd_net"] * (1 + self.scatter * rngdata[2])
                indices[name] = detindx

            k2 = None if self.vary else 0
            if self.pairs:
                # iterate over pairs of detectors
                for row1, row2 in pairwise(fp_data):
                    _process_row(row1, key2=k2)
                    _process_row(row2, key2=k2)
            else:
                # iterate over single detectors
                for row in fp_data:
                    _process_row(row, key2=k2)

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
