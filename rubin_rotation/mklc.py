import numpy as np
import scipy
import numpy
from .inject import generate_visits

def mklc(t, nspot=200, incl=np.pi*5./12., amp=1., tau=30.5, p=10.0):

    diffrot = 0.

    ''' This is a simplified version of the class-based routines in
    spot_model.py. It generates a light curves for dark, point like
    spots with no limb-darkening.

    Parameters:
    nspot = desired number of spots present on star at any
            one time
    amp = desired light curve amplitude
    tau = characteristic spot life-time
    diffrot = fractional difference between equatorial and polar
              rotation period
    (unit of time is equatorial rotation period)'''

    # print('Period = ', p)
    dur = (max(t) - min(t))

    # (crude estimate of) total number of spots needed during entire
    # time-series
    nspot_tot = int(nspot * dur / 2 / tau)

    # uniform distribution of spot longitudes
    lon = scipy.rand(nspot_tot) * 2 * np.pi

   # distribution of spot latitudes uniform in sin(latitude)
    lat = np.arcsin(scipy.rand(nspot_tot))

    # spot rotation rate optionally depends on latitude
    period = ((np.sin(lat) - 0.5) * diffrot + 1.0 ) * p
    period0 = scipy.ones(nspot_tot) * p

    # all spots have the same maximum area
    # (crude estimate of) filling factor needed per spot
    ff = amp / np.sqrt(nspot)
    scale_fac = 1
    amax = scipy.ones(nspot_tot) * ff * scale_fac

    # all spots have the evolution timescale
    decay = scipy.ones(nspot_tot) * tau

    # uniform distribution of spot peak times
    # start well before and end well after time-series limits (to
    # avoid edge effects)
    extra = 3 * decay.max()
    pk = scipy.rand(nspot_tot) * (dur + 2 * extra) - extra

    # COMPUTE THE LIGHT CURVE
    # print("Computing light curve...")
    time = numpy.array(t - min(t))

    area_tot = scipy.zeros_like(time)
    dF_tot = scipy.zeros_like(time)
    dF_tot0 = scipy.zeros_like(time)

    # add up the contributions of individual spots
    for i in range(nspot_tot):

        # Spot area
        if (pk[i] == 0) + (decay[i] == 0):
            area = scipy.ones_like(time) * amax[i]
        else:
            area = amax[i] * \
                scipy.exp(-(time - pk[i])**2 / 2. / decay[i]**2)
        area_tot += area

        # Fore-shortening
        phase = 2 * np.pi * time / period[i] + lon[i]
        phase0 = 2 * np.pi * time / period0[i] + lon[i]
        mu = np.cos(incl) * np.sin(lat[i]) + \
            np.sin(incl) * np.cos(lat[i]) * np.cos(phase)
        mu0 = np.cos(incl) * np.sin(lat[i]) + \
            np.sin(incl) * np.cos(lat[i]) * np.cos(phase0)
        mu[mu < 0] = 0.0
        mu0[mu0 < 0] = 0.0

        # Flux
        dF_tot -= area * mu
        dF_tot0 -= area * mu0

    amp_eff = dF_tot.max()-dF_tot.min()
    nspot_eff = area_tot / scale_fac / ff

    res0 = np.array([nspot_eff.mean(), ff, amp_eff])
    res1 = scipy.zeros((4, len(time)))

    res1[0,:] = time
    res1[1,:] = area_tot
    res1[2,:] = dF_tot
    res1[3,:] = dF_tot0

    # print('Used %d spots in total over %d rotation periods.' % (nspot_tot, dur))
    # print('Mean filling factor of individual spots was %.4f.' % ff)
    # print('Desired amplitude was %.4f, actual amplitude was %.4f.' \
    #         % (amp, amp_eff))
    # print('Desired number of spots at any one time was %d.' % nspot)
    return res0, res1


def sim_lc(prot, err, Nvisits=80, tspan=1, seed=42, tau_range=(1, 3)):

    np.random.seed(seed)

    time = generate_visits(Nvisits=Nvisits, tspan=tspan)

    sin2incl = np.random.uniform(np.sin(0)**2, np.sin(np.pi/2)**2)
    incl = np.arcsin(sin2incl**.5)
    tau = np.exp(np.random.uniform(np.log(tau_range[0]*prot), np.log(tau_range[1]*prot)))

    # Get LC
    res0, res1 = mklc(time, incl=incl, tau=tau, p=prot)
    nspot, ff, amp_err = res0
    _, area_tot, dF_tot, dF_tot0 = res1
    pure_flux = dF_tot0 / np.median(dF_tot0) - 1

    flux = pure_flux + np.random.randn(Nvisits) * err
    flux_err = np.ones_like(flux)*err

    return time, flux, pure_flux, flux_err
