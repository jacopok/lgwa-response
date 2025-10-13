from lgwa_response.likelihood import LunarLikelihood


def plot_mollview(time, psi):
    like = LunarLikelihood()

    like.get_antenna_response(time, ra, dec, psi)
