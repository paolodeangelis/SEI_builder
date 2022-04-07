pair_style_control = dict(
    settings="reaxff",
    file_path="control.pair_style",
    simulation_name="reax.output",  # output files will carry this name + their specific ext
    tabulate_long_range=10000,  # denotes the granularity of long range tabulation, 0 means no tabulation
    energy_update_freq=0,
    nbrhood_cutoff=5.0,  # near neighbors cutoff for bond calculations in A
    hbond_cutoff=7.5,  # cutoff distance for hydrogen bond interactions
    bond_graph_cutoff=0.35,  # bond strength cutoff for bond graphs
    thb_cutoff=0.001,  # cutoff value for three body interactions
    write_freq=0,  # write trajectory after so many steps
    traj_title="REAXFF_SIM",  # (no white spaces)
    atom_info=0,  # 0: no atom info, 1: print basic atom info in the trajectory file
    atom_forces=0,  # 0: basic atom format, 1: print force on each atom in the trajectory file
    atom_velocities=0,  # 0: basic atom format, 1: print the velocity of each atom in the trajectory file
    bond_info=0,  # 0: do not print bonds, 1: print bonds in the trajectory file
    angle_info=0,  # 0: do not print angles, 1: print angles in the trajectory file
)


# reax energy  list
reax_energy = dict(
    eb=1,  # bond energy
    ea=2,  # atom energy
    elp=3,  # lone-pair energy
    emol=4,  # molecule energy
    ev=5,  # valence angle energy
    epen=6,  # double-bond valence angle penalty
    ecoa=7,  # valence angle conjugation energy
    ehb=8,  # hydrogen bond energy
    et=9,  # torsion energy
    eco=10,  # conjugation energy
    ew=11,  # van der Waals energy
    ep=12,  # Coulomb energy
    efi=13,  # electric field energy
    eqeq=14,  # charge equilibration energy
)

default_em = dict(
    units="real",
    atom_style="full",
    read_data="data.system",
    pair_style=pair_style_control,
    pair_coeff="UNKNOW",
    fix=["1 all qeq/reax 1 0.0 10.0 1.0e-6 reax/c"],
    neighbor="1.0 bin",
    neigh_modify="every 10 delay 0 check yes",
    thermo=5,
    thermo_style="custom step temp pe etotal press vol fmax fnorm ",
    dump="1 all atom 100 dump.atom",
    # timestep=0.001,
    minimize="0.0 50 1000 3000",
    min_style="fire",
)
