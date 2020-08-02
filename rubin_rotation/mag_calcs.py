import numpy as np
from isochrones.mist.bc import MISTBolometricCorrectionGrid
bc_grid = MISTBolometricCorrectionGrid(['u', 'g', 'r', 'i', 'z', 'V'])
from isochrones import get_ichrone
mist = get_ichrone('mist')

def get_mag_from_mass(mass, age_gyr, feh):
    log_age = np.log10(age_gyr*1e9)
    eep = mist.get_eep(mass, age_gyr, feh, accurate=True)
    teff, logg = mist.interp_value([eep, age_gyr, feh], ['Teff', 'logg'])
    mags = bc_grid.interp([teff, logg, feh, 0.],
                          ['u', 'g', 'r', 'i', 'z', 'V'])
    rmag = mags[2]
    return teff, rmag

def calc_distance_from_M(M, m=24):
    # m - M = 5log(D) - 5
    D = 10**((m - M + 5)/5)
    return D*1e-3


if __name__ == "__main__":
    lim = 15
    faint_lim = 25

    print(get_mag_from_mass(1., 4.56, 0.))
    assert 0

    print(calc_distance_from_M(4.76, m=lim), "-",
          calc_distance_from_M(4.76, m=faint_lim), "kpc, G")  # Sun

    print(calc_distance_from_M(8.91, m=lim), "-",
          calc_distance_from_M(8.91, m=faint_lim), "kpc, M0")  # M0

    print(calc_distance_from_M(11.02, m=lim), "-",
          calc_distance_from_M(11.02, m=faint_lim), "kpc, M5")  # M5

    print(calc_distance_from_M(13.62, m=lim)*1e3, "-",
          calc_distance_from_M(13.62, m=faint_lim)*1e3, "pc, M8")  # M8
