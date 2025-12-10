from wrf_massive.base import Simulation

# This is a config dictionary, which will be inserted into the namelist.tmpl.input (template)
mynn25 = {
    # MYNN 2.5
    "physics__sf_sfclay_physics": "5",
    "physics__bl_pbl_physics": "5",
    "physics__bl_mynn_closure": "2.5",
    # without EDMF (so, mixing length 1)
    "physics__bl_mynn_mixlength": "1",
    "physics__bl_mynn_edmf": "0",
    "physics__bl_mynn_edmf_mom": "0",
    "physics__bl_mynn_edmf_tke": "0",
    "physics__sf_urban_physics": "0",  # no urban model
}


# Simple test simulation with 1-day usable data.
# 12h warmup is added automatically to the beginning.
sim_test = Simulation(
    begin="2020-01-01T00:00:00",
    end="2020-01-02T00:00:00",
    warmup_h=12,
    sim_dir="test_1",
    settings=mynn25,  # specify settings for namelist here
)


if __name__ == "__main__":
    # Just printing the simulation object shows all settings
    print(sim_test)
