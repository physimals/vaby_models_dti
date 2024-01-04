"""
VABY_MODELS_DTI - Ball / stick model
"""
import tensorflow as tf
import numpy as np

from vaby.fwd_model import FwdModel, FwdModelOption
from vaby.utils import ValueList, Matrix, NP_DTYPE, TF_DTYPE
from vaby.parameter import Parameter

from . import __version__

class BallStickModel(FwdModel):
    """
    DTI ball and stick model
    """

    @property
    def options(self):
        return [
            FwdModelOption("bvals", "B values", type=ValueList(float)),
            FwdModelOption("bvecs", "B vectors", type=Matrix),
            FwdModelOption("num_sticks", "Number of sticks to model anisotropy", type=int, default=3),
        ]

    def __init__(self, structure, ntpts, **options):
        FwdModel.__init__(self, structure, ntpts, **options)

        if self.num_sticks <= 0:
            raise ValueError(f"Number of sticks must be > 0: {self.num_sticks}")

        if self.bvals is None or self.bvecs is None:
            raise ValueError("BVALS and BVECS must be provided")
        self.bvals = np.array(self.bvals)
        if self.bvecs.ndim != 2 or self.bvecs.shape[1] != 3:
            raise ValueError(f"BVECS, expected n x 3 matrix, got {self.bvecs.shape}")
        if len(self.bvecs) != len(self.bvals):
            raise ValueError(f"Must have same number of BVALS and BVECS: got {len(self.bvals)}, {len(self.bvecs)}")

        self.params.append(
            Parameter(
                self, "s0", desc="Base signal offset",
                prior_mean=0, prior_var=1e6, post_mean=self._init_s0,
                **options
            )
        )
        self.params.append(
            Parameter(
                self, "d", desc="d",
                prior_mean=0, prior_var=1e6,
                **options
            )
        )
        for idx in range(self.num_sticks):
            self.params.append(
                Parameter(
                    self, f"theta{idx}", desc=f"Azimuth angle for stick {idx}",
                    prior_mean=0, prior_var=1e6,
                    **options
                )
            )
            self.params.append(
                Parameter(
                    self, f"phi{idx}", desc=f"Elevation angle for stick {idx}",
                    prior_mean=0, prior_var=1e6,
                    **options
                )
            )
            self.params.append(
                Parameter(
                    self, f"f{idx}", desc=f"Relative volume fraction for stick {idx}",
                    prior_mean=0, prior_var=1e6,
                    **options
                )
            )

    def __str__(self):
        return "DTI ball-and-stick model: %s" % __version__

    def evaluate(self, params, tpts):
        """
        DTI ball and stick model

        :param t: Time values tensor of shape [W, 1, N] or [1, 1, N]
        :param params Sequence of parameter values arrays, one for each parameter.
                      Each array is [W, S, 1] tensor where W is the number of nodes and
                      S the number of samples. This
                      may be supplied as a [P, W, S, 1] tensor where P is the number of
                      parameters.

        :return: [W, S, N] tensor containing model output at the specified time values
                 and for each time value using the specified parameter values
        """
        # FIXME f -> fractional 0-1 value?
        # FIXME how to constrain f so sum is < 1? or just ignore
        s0 = params[0]  # [W, S, 1]
        d = params[1]  # [W, S, 1]

        bvals_d = -self.bvals[None, None, :] * d  # [1, 1, N]

        f = tf.ones(s0.shape, dtype=TF_DTYPE)  # [W, S, 1]
        sum_f = tf.zeros(s0.shape, dtype=TF_DTYPE)  # [W, S, 1]
        signal = None
        for idx in range(self.num_sticks):
            v = self._get_vector(params[2 + 3 * idx], params[3 + 3 * idx])  # [3, W, S, 1]

            # This formula reduces f to a number between 0 and 1
            fr = tf.math.reciprocal_no_nan(tf.exp(params[4 + 3 * idx]))  # [W, S, 1]
            f *= fr  # [W, S, 1]
            stick_signal = fr * np.exp(bvals_d * np.square(np.dot(self.bvecs, v)))  # [W, S, N]
            if signal is None:
                signal = stick_signal
            else:
                signal += stick_signal
            sum_f += fr

        signal = s0 * (1 - sum_f) * np.exp(bvals_d)  # [W, S, N]
        return signal

    def _get_vector(self, theta, phi):
        """
        Get a 3D unit vector from azimuth and elevation angles

        The angle values are in the range -inf, inf so must first
        be scaled to the appropriate ranges for spherical co-ordinates

        :param theta: Tensor containing azimuth values
        :param phi: Tensor containing elevation values
        """
        raise NotImplementedError()
