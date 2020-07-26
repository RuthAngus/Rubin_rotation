import numpy as np

import exoplanet as xo
import pymc3 as pm
import theano.tensor as tt


def multiband_gp(x, y, yerr, inds, peak):

    with pm.Model() as model:

        # The mean flux of the time series
        mean = pm.Normal("mean", mu=0.0, sd=10.0)

        # The log-amplitude of the r-band light curve, relative to the g-band
        log_ramp = pm.Normal("log_ramp", mu=-.7, sd=2.)
        y_tensor = y * (1 - (1 - tt.exp(-log_ramp)) * inds)
        y_err_tensor = yerr * (1 - (1 - tt.exp(-log_ramp)) * inds)
    #     y[inds] *= tt.exp(log_ramp)
    #     yerr[inds] *= tt.exp(log_ramp)

        # A jitter term describing excess white noise
        logs2 = pm.Normal("logs2", mu=2 * np.log(np.mean(yerr)), sd=2.0)

        # A term to describe the non-periodic variability
        logSw4 = pm.Normal("logSw4", mu=np.log(np.var(y)), sd=5.0)
        logw0 = pm.Normal("logw0", mu=np.log(2 * np.pi / 10), sd=5.0)

        # The parameters of the RotationTerm kernel
        logamp = pm.Normal("logamp", mu=np.log(np.var(y)), sd=5.0)
        BoundedNormal = pm.Bound(pm.Normal, lower=0.0, upper=np.log(50))
        logperiod = BoundedNormal("logperiod", mu=np.log(peak["period"]), sd=5.0)
        logQ0 = pm.Normal("logQ0", mu=1.0, sd=10.0)
        logdeltaQ = pm.Normal("logdeltaQ", mu=2.0, sd=10.0)
        mix = xo.distributions.UnitUniform("mix")

        # Track the period as a deterministic
        period = pm.Deterministic("period", tt.exp(logperiod))

        # Set up the Gaussian Process model
        kernel = xo.gp.terms.SHOTerm(log_Sw4=logSw4, log_w0=logw0, Q=1 / np.sqrt(2))
        kernel += xo.gp.terms.RotationTerm(
            log_amp=logamp, period=period, log_Q0=logQ0, log_deltaQ=logdeltaQ, mix=mix
        )
        gp = xo.gp.GP(kernel, x, y_err_tensor ** 2 + tt.exp(logs2), mean=mean)

        # Compute the Gaussian Process likelihood and add it into the
        # the PyMC3 model as a "potential"
        gp.marginal("gp", observed=y_tensor)

        # Compute the mean model prediction for plotting purposes
        pm.Deterministic("pred", gp.predict())

        # Optimize to find the maximum a posteriori parameters
        map_soln = xo.optimize(start=model.test_point)

    return map_soln
