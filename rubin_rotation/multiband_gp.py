import numpy as np

import exoplanet as xo
import pymc3 as pm
import theano.tensor as tt


def make_data_nice(x, y, yerr):
    sort = np.argsort(x)
    y = np.ascontiguousarray(y[sort], dtype=np.float64)
    yerr = np.ascontiguousarray(yerr[sort], dtype=np.float64)
    x = np.ascontiguousarray(x[sort], dtype=np.float64)
    return x, y, yerr


class Star(object):

    def __init__(self, x, y, yerr, init_period):
        self.x = x
        self.y = y
        self.yerr = yerr
        self.init_period = init_period

    def multiband_gp(self, inds, seed=42, lower=.1, upper=150):
        # x, y, yerr = make_data_nice(x, y, yerr)
        np.random.seed(seed)

        with pm.Model() as model:

            # The mean flux of the time series
            mean = pm.Normal("mean", mu=0.0, sd=10.0)
            meanr = pm.Normal("meanr", mu=0.0, sd=10.0)

            # The log-amplitude of the r-band light curve, relative to the g-band
            log_ramp = pm.Normal("log_ramp", mu=-.7, sd=2.)
            # y_tensor = y * (1 - (1 - tt.exp(-log_ramp)) * inds)
            y_tensor = (self.y - meanr*inds) * (1 - (1 - tt.exp(-log_ramp)) * inds)
            y_err_tensor = self.yerr * (1 - (1 - tt.exp(-log_ramp)) * inds)

            # A jitter term describing excess white noise
            logs2 = pm.Normal("logs2", mu=2 * np.log(np.mean(self.yerr)), sd=2.0)

            # A term to describe the non-periodic variability
            logSw4 = pm.Normal("logSw4", mu=np.log(np.var(self.y)), sd=5.0)
            logw0 = pm.Normal("logw0", mu=np.log(2 * np.pi / 10), sd=5.0)

            # The parameters of the RotationTerm kernel
            logamp = pm.Normal("logamp", mu=np.log(np.var(self.y)), sd=5.0)
            BoundedNormal = pm.Bound(pm.Normal, lower=lower, upper=np.log(upper))
            logperiod = BoundedNormal("logperiod", mu=np.log(self.init_period), sd=5.0)
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
            gp = xo.gp.GP(kernel, self.x, y_err_tensor ** 2 + tt.exp(logs2), mean=mean)

            # Compute the Gaussian Process likelihood and add it into the
            # the PyMC3 model as a "potential"
            gp.marginal("gp", observed=y_tensor)

            # Compute the mean model prediction for plotting purposes
            pm.Deterministic("pred", gp.predict())

            # Optimize to find the maximum a posteriori parameters
            map_soln = xo.optimize(start=model.test_point)
            self.map_soln = map_soln

        self.model = model
        self.map_soln = model
        return map_soln, model

    def singleband_gp(self, lower=5, upper=50, seed=42):
    # x, y, yerr = make_data_nice(x, y, yerr)
        np.random.seed(seed)

        with pm.Model() as model:

            # The mean flux of the time series
            mean = pm.Normal("mean", mu=0.0, sd=10.0)

            # A jitter term describing excess white noise
            logs2 = pm.Normal("logs2", mu=2 * np.log(np.mean(self.yerr)), sd=2.0)

            # A term to describe the non-periodic variability
            logSw4 = pm.Normal("logSw4", mu=np.log(np.var(self.y)), sd=5.0)
            logw0 = pm.Normal("logw0", mu=np.log(2 * np.pi / 10), sd=5.0)

            # The parameters of the RotationTerm kernel
            logamp = pm.Normal("logamp", mu=np.log(np.var(self.y)), sd=5.0)
            BoundedNormal = pm.Bound(pm.Normal, lower=np.log(lower),
                                    upper=np.log(upper))
            logperiod = BoundedNormal("logperiod", mu=np.log(self.init_period),
                                    sd=5.0)
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
            gp = xo.gp.GP(kernel, self.x, self.yerr ** 2 + tt.exp(logs2), mean=mean)

            # Compute the Gaussian Process likelihood and add it into the
            # the PyMC3 model as a "potential"
            gp.marginal("gp", observed=self.y)

        # Compute the mean model prediction for plotting purposes
            pm.Deterministic("pred", gp.predict())

            # Optimize to find the maximum a posteriori parameters
            map_soln = xo.optimize(start=model.test_point)

        self.model = model
        self.map_soln = model
        return map_soln, model

    # def mcmc(self):
    #     with self.model:
    #         trace = xo.sample(tune=500, draws=1000, start=self.map_soln,
    #                         target_accept=0.95)
