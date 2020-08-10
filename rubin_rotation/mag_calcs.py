import numpy as np
from isochrones.mist.bc import MISTBolometricCorrectionGrid
bc_grid = MISTBolometricCorrectionGrid(['u', 'g', 'r', 'i', 'z', 'V'])
from isochrones import get_ichrone
mist = get_ichrone('mist')

def get_mag_from_mass(mass, age_gyr, feh):
    log_age = np.log10(age_gyr*1e9)
    eep = mist.get_eep(mass, log_age, feh, accurate=True)
    teff, logg = mist.interp_value([eep, log_age, feh], ['Teff', 'logg'])
    mags = bc_grid.interp([teff, logg, feh, 0.],
                          ['u', 'g', 'r', 'i', 'z', 'V'])
    rmag = mags[2]
    return teff, rmag

def calc_distance_from_M(M, m=24):
    # m - M = 5log(D) - 5
    D = 10**((m - M + 5)/5)
    return D*1e-3


if __name__ == "__main__":
    lim = 16
    faint_lim = 24
    ztf = True
    f = 1
    if ztf:
        lim = 13  # ZTF
        faint_lim = 17 # ZTF
        f = 1e3

    # teff, rmag = get_mag_from_mass(1., 4.56, 0.)
    # print(calc_distance_from_M(rmag, m=lim), "-",
    #       calc_distance_from_M(rmag, m=faint_lim), "kpc, G", rmag)  # Sun

    # teff, rmag = get_mag_from_mass(.55, 12, 0.)
    # print(calc_distance_from_M(rmag, m=lim), "-",
    #       calc_distance_from_M(rmag, m=faint_lim), "kpc, M0", rmag)  # M0

    # teff, rmag = get_mag_from_mass(.16, 12, 0.)
    # print(calc_distance_from_M(rmag, m=lim), "-",
    #       calc_distance_from_M(rmag, m=faint_lim), "kpc, M5", rmag)  # M5

    # teff, rmag = get_mag_from_mass(.082, 12, 0.)
    # print(calc_distance_from_M(rmag, m=lim)*1e3, "-",
    #       calc_distance_from_M(rmag, m=faint_lim)*1e3, "pc, M8", rmag)  # M8

    print(calc_distance_from_M(0, m=lim)*f, "-",
          calc_distance_from_M(0, m=faint_lim)*f, "kpc, subgiant")  # Sun

    print(calc_distance_from_M(3.87, m=lim)*f, "-",
          calc_distance_from_M(3.87, m=faint_lim)*f, "kpc, F7")  # F0

    print(calc_distance_from_M(4.45, m=lim)*f, "-",
          calc_distance_from_M(4.45, m=faint_lim)*f, "kpc, G0")  # Sun

    print(calc_distance_from_M(5.55, m=lim)*f, "-",
          calc_distance_from_M(5.55, m=faint_lim)*f, "kpc, G9")  # Sun

    print(calc_distance_from_M(5.76, m=lim)*f, "-",
          calc_distance_from_M(5.76, m=faint_lim)*f, "kpc, K0")  # Ko

    print(calc_distance_from_M(8.69, m=lim)*f, "-",
          calc_distance_from_M(8.69, m=faint_lim)*f, "kpc, K9")  # Ko

    print(calc_distance_from_M(8.91, m=lim)*f, "-",
          calc_distance_from_M(8.91, m=faint_lim)*f, "kpc, M0")  # M0

    print(calc_distance_from_M(11.02, m=lim)*f, "-",
          calc_distance_from_M(11.02, m=faint_lim)*f, "kpc, M5")  # M5

    print(calc_distance_from_M(13.62, m=lim)*f, "-",
          calc_distance_from_M(13.62, m=faint_lim)*f, "kpc, M8")  # M8
