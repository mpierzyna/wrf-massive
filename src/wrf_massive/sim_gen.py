import datetime
from wrf_massive.base import Simulation


def get_spaced_sims(
    global_begin: datetime.datetime,
    global_end: datetime.datetime,
    settings: dict[str, str],
    net_sim_days: int = 5,
    step: int = 4,
    offset: int = 0,
) -> list[Simulation]:
    """Generate a list of simulations spaced over a time ranged
    
    Parameters
    ----------
    global_begin : datetime.datetime
        The start of the time range to generate simulations for.
    global_end : datetime.datetime
        The end of the time range to generate simulations for.
    settings : dict[str, str]
        The settings to use for each simulation.
    net_sim_days : int, optional
        The number of days of usable data for each simulation, by default 5 days.
    step : int, optional
        Number of simulations to skip between each simulation, by default 4.
        So, (net_sim_days * step) number of days are skipped.
    offset : int, optional
        Number of simulations to skip at the beginning of the time range, by default 0.
        So, (offset * step) number of days are skipped at the beginning of the time range.

    Returns
    -------
    list[Simulation]
        A list of simulations.
    """
    global_days = (global_end - global_begin).days

    sims = []
    for i in range(offset * step, global_days, net_sim_days * step):
        begin = global_begin + datetime.timedelta(days=i)
        end = begin + datetime.timedelta(days=net_sim_days)
        sim = Simulation(
            begin=begin,
            end=end,
            warmup_h=12,
            sim_dir=f"sim_{begin.strftime('%Y-%m-%d')}",
            settings=settings,
        )
        sims.append(sim)
    return sims
