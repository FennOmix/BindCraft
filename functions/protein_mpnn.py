from colabdesign import clear_mem
from colabdesign.mpnn import mk_mpnn_model


def mpnn_gen_sequence(trajectory_pdb, binder_chain, trajectory_interface_residues, advanced_settings):
    """Generate MPNN sequences for a given binder trajectory."""
    clear_mem()

    mpnn_model = mk_mpnn_model(backbone_noise=advanced_settings["backbone_noise"],
                               model_name=advanced_settings["model_path"],
                               weights=advanced_settings["mpnn_weights"])

    design_chains = 'A,' + binder_chain

    if advanced_settings["mpnn_fix_interface"]:
        fixed_positions = 'A,' + trajectory_interface_residues
        fixed_positions = fixed_positions.rstrip(",")
        print("Fixing interface residues: " + trajectory_interface_residues)
    else:
        fixed_positions = 'A'

    mpnn_model.prep_inputs(pdb_filename=trajectory_pdb, chain=design_chains,
                           fix_pos=fixed_positions, rm_aa=advanced_settings["omit_AAs"])

    mpnn_sequences = mpnn_model.sample(temperature=advanced_settings["sampling_temp"],
                                       num=1, batch=advanced_settings["num_seqs"])

    return mpnn_sequences
