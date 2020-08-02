import numpy as np
import starry

# Define our spatial power spectrum
def power(l, amp=1e-1):
    return amp * np.exp(-((l / 10) ** 2))


def get_random_light_curve(t, p, a, inclination=90., seed=None):
    """
    Generate a random starry light curve

    Args:
        t (array): The time array
        p (float): The rotation period
        a (float) The amplitude of the signal
        inclination (Optional, float): The inclination of the star in degrees.
             Default=90.
        seed (Optional int): The random number seed.

    Returns:
        flux (array): The flux array.
    """

    if seed is not None:
        np.random.seed(seed)

    starry.config.lazy = False

    # Instantiate a 10th degree starry map
    map = starry.Map(10)

    # Random inclination (isotropically distributed ang. mom. vector)
    if inclination == "random":
        map.inc = np.arccos(np.random.random()) * 180 / np.pi
    else:
        map.inc = inclination

    # Random period, U[1, 30]
    # p = 1 + 29 * np.random.random()

    # Random surface map
    for l in range(1, map.ydeg + 1):
        map[l, :] = np.random.randn(2 * l + 1) * power(l, amp=a) / (2 * l + 1)

    # Compute the flux
    flux = map.flux(theta=360.0 * t / p)

    # Median-correct it
    flux -= np.median(flux)
    flux += 1

    return flux


def generate_visits(Nvisits=900, tspan=10, stat=False,
                    seasonscale=365./5):
    '''
    From Jim :-).
    Use some very crude approximations for how visits will be spaced out:
    - Survey starts at midnight, time = 0.0
    - Can only observe at night, time > 0.75 | time < 0.25
    - Exposures are clustered around a season w/ a gaussian shape each year
    - Field is observable for first half of year, 0 < date < 182
    - On average, each field should be hit every 3 days during observable season
    Set "stat=True" if you want a plot and a couple basic statistics about the cadence
    '''
    # generate random times for visit, between [0.75 and 0.25]
    time_of_day = np.random.random(Nvisits)/2. - 0.25

    date_of_year = np.floor(np.random.normal(loc=365./4., scale=seasonscale, size=Nvisits))

    year_of_obs = np.floor(np.random.random(Nvisits) * tspan) * 365.

    date_obs = time_of_day + date_of_year + year_of_obs

    date_obs.sort()

    if stat is True:
        print('mean time between visits:')
        print(np.mean(date_obs[1:] - date_obs[:-1]))

        print('median time between visits:')
        print(np.median(date_obs[1:] - date_obs[:-1]))

        plt.figure()
        _ = plt.hist(date_obs, bins=np.arange(date_obs.min(), date_obs.max(),7),
                     histtype='stepfilled', color='k')
        plt.xlabel('Time (days)')
        plt.ylabel('# Visits per Week')
        plt.show()

    return date_obs


def LSST_sig(m):
    """
    Approximate the noise in figure 2 of arxiv:1603.06638 from the apparent
    r-mag.

    Args:
        m (float): r-band apparant mag.
    Returns:
        white noise in magnitude dex.

    """

    if m < 19:
        return .005
    mags = np.array([19, 20, 21, 22, 23, 24, 25])
    sigs = np.array([.005, .007, .01, .02, .03, .1, .2])
    return sigs[np.abs(mags - m) == np.abs(mags-m).min()][0]
