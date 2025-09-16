"""Core execution pipeline for BindCraft binder design."""

from colabdesign.shared.utils import copy_dict
from colabdesign import mk_afdesign_model, clear_mem

import gc
import os
import shutil
import time
from typing import Dict

import numpy as np
import pandas as pd

from .design import binder_hallucination
from .protein_mpnn import mpnn_gen_sequence
from .alphafold2_validation import predict_binder_complex, predict_binder_alone
from .generic_utils import (
    check_jax_gpu,
    load_json_settings,
    load_af2_models,
    perform_advanced_settings_check,
    generate_directories,
    generate_dataframe_labels,
    create_dataframe,
    generate_filter_pass_csv,
    check_accepted_designs,
    check_n_trajectories,
    load_helicity,
    insert_data,
    save_fasta,
    validate_design_sequence,
    calculate_averages,
    check_filters,
)
from .pyrosetta_utils import pr, pr_relax, score_interface, unaligned_rmsd
from .biopython_utils import calc_ss_percentage, calculate_clash_score, target_pdb_rmsd


def design_binders(
    design_name: str,
    target_settings: Dict,
    length: int,
    seed: int,
    helicity_value: float,
    design_models,
    advanced_settings: Dict,
    design_paths: Dict[str, str],
    failure_csv: str,
):
    """Run binder hallucination for a single trajectory."""

    trajectory = binder_hallucination(
        design_name,
        target_settings["starting_pdb"],
        target_settings["chains"],
        target_settings["target_hotspot_residues"],
        length,
        seed,
        helicity_value,
        design_models,
        advanced_settings,
        design_paths,
        failure_csv,
    )

    trajectory_metrics = copy_dict(trajectory._tmp["best"]["aux"]["log"])
    trajectory_pdb = os.path.join(design_paths["Trajectory"], design_name + ".pdb")

    return trajectory, trajectory_metrics, trajectory_pdb


def generate_sequences_with_mpnn(
    trajectory_pdb: str,
    binder_chain: str,
    trajectory_interface_residues: str,
    advanced_settings: Dict,
):
    """Generate sequences for a trajectory using ProteinMPNN."""

    return mpnn_gen_sequence(
        trajectory_pdb, binder_chain, trajectory_interface_residues, advanced_settings
    )


def validate_with_alphafold2(
    sequence: str,
    design_name: str,
    target_settings: Dict,
    length: int,
    trajectory_pdb: str,
    prediction_models,
    advanced_settings: Dict,
    filters: Dict,
    design_paths: Dict[str, str],
    failure_csv: str,
):
    """Validate a sequence using AlphaFold2 multimer predictions."""

    return predict_binder_complex(
        mk_afdesign_model(
            protocol="binder",
            num_recycles=advanced_settings["num_recycles_validation"],
            data_dir=advanced_settings["af_params_dir"],
            use_multimer=True,
            use_initial_guess=advanced_settings["predict_initial_guess"],
            use_initial_atom_pos=advanced_settings["predict_bigbang"],
        ),
        sequence,
        design_name,
        target_settings["starting_pdb"],
        target_settings["chains"],
        length,
        trajectory_pdb,
        prediction_models,
        advanced_settings,
        filters,
        design_paths,
        failure_csv,
    )


def run_bindcraft(settings_path: str, filters_path: str, advanced_path: str) -> None:
    """Run the complete BindCraft binder design workflow."""

    # Ensure a JAX-capable GPU is available
    check_jax_gpu()

    # Load settings from JSON files
    target_settings, advanced_settings, filters = load_json_settings(
        settings_path, filters_path, advanced_path
    )

    settings_file = os.path.basename(settings_path).split(".")[0]
    filters_file = os.path.basename(filters_path).split(".")[0]
    advanced_file = os.path.basename(advanced_path).split(".")[0]

    # Load AF2 model settings
    design_models, prediction_models, multimer_validation = load_af2_models(
        advanced_settings["use_multimer_design"]
    )

    # Perform checks on advanced settings
    bindcraft_folder = os.path.dirname(
        os.path.dirname(os.path.realpath(__file__))
    )
    advanced_settings = perform_advanced_settings_check(
        advanced_settings, bindcraft_folder
    )

    # Generate directories
    design_paths = generate_directories(target_settings["design_path"])

    # Generate dataframes
    trajectory_labels, design_labels, final_labels = generate_dataframe_labels()

    trajectory_csv = os.path.join(
        target_settings["design_path"], "trajectory_stats.csv"
    )
    mpnn_csv = os.path.join(
        target_settings["design_path"], "mpnn_design_stats.csv"
    )
    final_csv = os.path.join(
        target_settings["design_path"], "final_design_stats.csv"
    )
    failure_csv = os.path.join(
        target_settings["design_path"], "failure_csv.csv"
    )

    create_dataframe(trajectory_csv, trajectory_labels)
    create_dataframe(mpnn_csv, design_labels)
    create_dataframe(final_csv, final_labels)
    generate_filter_pass_csv(failure_csv, filters_path)

    # Initialise PyRosetta
    pr.init(
        f'-ignore_unrecognized_res -ignore_zero_occupancy -mute all '
        f'-holes:dalphaball {advanced_settings["dalphaball_path"]} '
        f'-corrections::beta_nov16 true -relax:default_repeats 1'
    )
    print(f"Running binder design for target {settings_file}")
    print(f"Design settings used: {advanced_file}")
    print(f"Filtering designs based on {filters_file}")

    # Initialise counters
    script_start_time = time.time()
    trajectory_n = 1
    accepted_designs = 0

    # Start design loop
    while True:
        # Check if target number of binders achieved
        final_designs_reached = check_accepted_designs(
            design_paths,
            mpnn_csv,
            final_labels,
            final_csv,
            advanced_settings,
            target_settings,
            design_labels,
        )

        if final_designs_reached:
            break

        # Check if maximum trajectories reached
        max_trajectories_reached = check_n_trajectories(
            design_paths, advanced_settings
        )
        if max_trajectories_reached:
            break

        # Initialise design
        trajectory_start_time = time.time()

        # Generate random seed
        seed = int(np.random.randint(0, high=999999, size=1, dtype=int)[0])

        # Sample binder design length
        samples = np.arange(
            min(target_settings["lengths"]),
            max(target_settings["lengths"]) + 1,
        )
        length = np.random.choice(samples)

        # Load desired helicity value
        helicity_value = load_helicity(advanced_settings)

        # Generate design name and check if trajectory already exists
        design_name = (
            target_settings["binder_name"] + "_l" + str(length) + "_s" + str(seed)
        )
        trajectory_dirs = [
            "Trajectory",
            "Trajectory/Relaxed",
            "Trajectory/LowConfidence",
            "Trajectory/Clashing",
        ]
        trajectory_exists = any(
            os.path.exists(
                os.path.join(design_paths[trajectory_dir], design_name + ".pdb")
            )
            for trajectory_dir in trajectory_dirs
        )

        if not trajectory_exists:
            print("Starting trajectory: " + design_name)

            # Begin binder hallucination
            trajectory, trajectory_metrics, trajectory_pdb = design_binders(
                design_name,
                target_settings,
                length,
                seed,
                helicity_value,
                design_models,
                advanced_settings,
                design_paths,
                failure_csv,
            )

            # Round metrics
            trajectory_metrics = {
                k: round(v, 2) if isinstance(v, float) else v
                for k, v in trajectory_metrics.items()
            }

            # Time trajectory
            trajectory_time = time.time() - trajectory_start_time
            trajectory_time_text = "%d hours, %d minutes, %d seconds" % (
                int(trajectory_time // 3600),
                int((trajectory_time % 3600) // 60),
                int(trajectory_time % 60),
            )
            print("Starting trajectory took: " + trajectory_time_text)
            print("")

            # Proceed if no trajectory termination signal
            if trajectory.aux["log"]["terminate"] == "":
                # Relax binder
                trajectory_relaxed = os.path.join(
                    design_paths["Trajectory/Relaxed"], design_name + ".pdb"
                )
                pr_relax(trajectory_pdb, trajectory_relaxed)

                # Binder chain placeholder
                binder_chain = "B"

                # Calculate clashes
                num_clashes_trajectory = calculate_clash_score(trajectory_pdb)
                num_clashes_relaxed = calculate_clash_score(trajectory_relaxed)

                # Secondary structure content
                (
                    trajectory_alpha,
                    trajectory_beta,
                    trajectory_loops,
                    trajectory_alpha_interface,
                    trajectory_beta_interface,
                    trajectory_loops_interface,
                    trajectory_i_plddt,
                    trajectory_ss_plddt,
                ) = calc_ss_percentage(
                    trajectory_pdb, advanced_settings, binder_chain
                )

                # Interface scores
                (
                    trajectory_interface_scores,
                    trajectory_interface_AA,
                    trajectory_interface_residues,
                ) = score_interface(trajectory_relaxed, binder_chain)

                # Starting binder sequence
                trajectory_sequence = trajectory.get_seq(get_best=True)[0]

                # Sequence analysis
                traj_seq_notes = validate_design_sequence(
                    trajectory_sequence, num_clashes_relaxed, advanced_settings
                )

                # Target structure RMSD compared to input PDB
                trajectory_target_rmsd = target_pdb_rmsd(
                    trajectory_pdb,
                    target_settings["starting_pdb"],
                    target_settings["chains"],
                )

                # Save trajectory statistics
                trajectory_data = [
                    design_name,
                    advanced_settings["design_algorithm"],
                    length,
                    seed,
                    helicity_value,
                    target_settings["target_hotspot_residues"],
                    trajectory_sequence,
                    trajectory_interface_residues,
                    trajectory_metrics["plddt"],
                    trajectory_metrics["ptm"],
                    trajectory_metrics["i_ptm"],
                    trajectory_metrics["pae"],
                    trajectory_metrics["i_pae"],
                    trajectory_i_plddt,
                    trajectory_ss_plddt,
                    num_clashes_trajectory,
                    num_clashes_relaxed,
                    trajectory_interface_scores["binder_score"],
                    trajectory_interface_scores["surface_hydrophobicity"],
                    trajectory_interface_scores["interface_sc"],
                    trajectory_interface_scores["interface_packstat"],
                    trajectory_interface_scores["interface_dG"],
                    trajectory_interface_scores["interface_dSASA"],
                    trajectory_interface_scores["interface_dG_SASA_ratio"],
                    trajectory_interface_scores["interface_fraction"],
                    trajectory_interface_scores["interface_hydrophobicity"],
                    trajectory_interface_scores["interface_nres"],
                    trajectory_interface_scores["interface_interface_hbonds"],
                    trajectory_interface_scores["interface_hbond_percentage"],
                    trajectory_interface_scores[
                        "interface_delta_unsat_hbonds"
                    ],
                    trajectory_interface_scores[
                        "interface_delta_unsat_hbonds_percentage"
                    ],
                    trajectory_alpha_interface,
                    trajectory_beta_interface,
                    trajectory_loops_interface,
                    trajectory_alpha,
                    trajectory_beta,
                    trajectory_loops,
                    trajectory_interface_AA,
                    trajectory_target_rmsd,
                    trajectory_time_text,
                    traj_seq_notes,
                    settings_file,
                    filters_file,
                    advanced_file,
                ]
                insert_data(trajectory_csv, trajectory_data)

                if not trajectory_interface_residues:
                    print(
                        "No interface residues found for "
                        + str(design_name)
                        + ", skipping MPNN optimization"
                    )
                    continue

                if advanced_settings["enable_mpnn"]:
                    # Initialise MPNN counters
                    mpnn_n = 1
                    accepted_mpnn = 0
                    mpnn_dict = {}
                    design_start_time = time.time()

                    # MPNN redesign
                    mpnn_trajectories = generate_sequences_with_mpnn(
                        trajectory_pdb,
                        binder_chain,
                        trajectory_interface_residues,
                        advanced_settings,
                    )
                    existing_mpnn_sequences = set(
                        pd.read_csv(mpnn_csv, usecols=["Sequence"])[
                            "Sequence"
                        ].values
                    )

                    # Create set of sequences with allowed amino acid composition
                    restricted_AAs = (
                        set(aa.strip().upper() for aa in advanced_settings["omit_AAs"].split(","))
                        if advanced_settings["force_reject_AA"]
                        else set()
                    )

                    mpnn_sequences = sorted(
                        {
                            mpnn_trajectories["seq"][n][-length:]: {
                                "seq": mpnn_trajectories["seq"][n][-length:],
                                "score": mpnn_trajectories["score"][n],
                                "seqid": mpnn_trajectories["seqid"][n],
                            }
                            for n in range(advanced_settings["num_seqs"])
                            if (
                                not restricted_AAs
                                or not any(
                                    aa in mpnn_trajectories["seq"][n][
                                        -length:
                                    ].upper()
                                    for aa in restricted_AAs
                                )
                            )
                            and mpnn_trajectories["seq"][n][-length:]
                            not in existing_mpnn_sequences
                        }.values(),
                        key=lambda x: x["score"],
                    )

                    del existing_mpnn_sequences

                    # Check whether any sequences left
                    if mpnn_sequences:
                        # Adjust recycles if trajectory is beta sheeted
                        if advanced_settings["optimise_beta"] and float(trajectory_beta) > 15:
                            advanced_settings["num_recycles_validation"] = advanced_settings[
                                "optimise_beta_recycles_valid"
                            ]

                        # Compile prediction models once
                        clear_mem()
                        complex_prediction_model = mk_afdesign_model(
                            protocol="binder",
                            num_recycles=advanced_settings["num_recycles_validation"],
                            data_dir=advanced_settings["af_params_dir"],
                            use_multimer=multimer_validation,
                            use_initial_guess=advanced_settings["predict_initial_guess"],
                            use_initial_atom_pos=advanced_settings["predict_bigbang"],
                        )
                        if advanced_settings["predict_initial_guess"] or advanced_settings["predict_bigbang"]:
                            complex_prediction_model.prep_inputs(
                                pdb_filename=trajectory_pdb,
                                chain="A",
                                binder_chain="B",
                                binder_len=length,
                                use_binder_template=True,
                                rm_target_seq=advanced_settings["rm_template_seq_predict"],
                                rm_target_sc=advanced_settings["rm_template_sc_predict"],
                                rm_template_ic=True,
                            )
                        else:
                            complex_prediction_model.prep_inputs(
                                pdb_filename=target_settings["starting_pdb"],
                                chain=target_settings["chains"],
                                binder_len=length,
                                rm_target_seq=advanced_settings["rm_template_seq_predict"],
                                rm_target_sc=advanced_settings["rm_template_sc_predict"],
                            )

                        # Compile binder monomer prediction model
                        binder_prediction_model = mk_afdesign_model(
                            protocol="hallucination",
                            use_templates=False,
                            initial_guess=False,
                            use_initial_atom_pos=False,
                            num_recycles=advanced_settings["num_recycles_validation"],
                            data_dir=advanced_settings["af_params_dir"],
                            use_multimer=multimer_validation,
                        )
                        binder_prediction_model.prep_inputs(length=length)

                        # Iterate over designed sequences
                        for mpnn_sequence in mpnn_sequences:
                            mpnn_time = time.time()

                            # Generate MPNN design name numbering
                            mpnn_design_name = design_name + "_mpnn" + str(mpnn_n)
                            mpnn_score = round(mpnn_sequence["score"], 2)
                            mpnn_seqid = round(mpnn_sequence["seqid"], 2)

                            # Add design to dictionary
                            mpnn_dict[mpnn_design_name] = {
                                "seq": mpnn_sequence["seq"],
                                "score": mpnn_score,
                                "seqid": mpnn_seqid,
                            }

                            # Save fasta sequence
                            if advanced_settings["save_mpnn_fasta"] is True:
                                save_fasta(
                                    mpnn_design_name, mpnn_sequence["seq"], design_paths
                                )

                            # Predict binder complex
                            (
                                mpnn_complex_statistics,
                                pass_af2_filters,
                            ) = validate_with_alphafold2(
                                mpnn_sequence["seq"],
                                mpnn_design_name,
                                target_settings,
                                length,
                                trajectory_pdb,
                                prediction_models,
                                advanced_settings,
                                filters,
                                design_paths,
                                failure_csv,
                            )

                            # Skip scoring if AF2 filters not passed
                            if not pass_af2_filters:
                                print(
                                    f"Base AF2 filters not passed for {mpnn_design_name}, skipping interface scoring"
                                )
                                mpnn_n += 1
                                continue

                            # Calculate statistics for each model individually
                            for model_num in prediction_models:
                                mpnn_design_pdb = os.path.join(
                                    design_paths["MPNN"],
                                    f"{mpnn_design_name}_model{model_num+1}.pdb",
                                )
                                mpnn_design_relaxed = os.path.join(
                                    design_paths["MPNN/Relaxed"],
                                    f"{mpnn_design_name}_model{model_num+1}.pdb",
                                )

                                if os.path.exists(mpnn_design_pdb):
                                    num_clashes_mpnn = calculate_clash_score(
                                        mpnn_design_pdb
                                    )
                                    num_clashes_mpnn_relaxed = calculate_clash_score(
                                        mpnn_design_relaxed
                                    )

                                    (
                                        mpnn_interface_scores,
                                        mpnn_interface_AA,
                                        mpnn_interface_residues,
                                    ) = score_interface(
                                        mpnn_design_relaxed, binder_chain
                                    )

                                    (
                                        mpnn_alpha,
                                        mpnn_beta,
                                        mpnn_loops,
                                        mpnn_alpha_interface,
                                        mpnn_beta_interface,
                                        mpnn_loops_interface,
                                        mpnn_i_plddt,
                                        mpnn_ss_plddt,
                                    ) = calc_ss_percentage(
                                        mpnn_design_pdb, advanced_settings, binder_chain
                                    )

                                    rmsd_site = unaligned_rmsd(
                                        trajectory_pdb,
                                        mpnn_design_pdb,
                                        binder_chain,
                                        binder_chain,
                                    )

                                    target_rmsd = target_pdb_rmsd(
                                        mpnn_design_pdb,
                                        target_settings["starting_pdb"],
                                        target_settings["chains"],
                                    )

                                    mpnn_complex_statistics[model_num + 1].update(
                                        {
                                            "i_pLDDT": mpnn_i_plddt,
                                            "ss_pLDDT": mpnn_ss_plddt,
                                            "Unrelaxed_Clashes": num_clashes_mpnn,
                                            "Relaxed_Clashes": num_clashes_mpnn_relaxed,
                                            "Binder_Energy_Score": mpnn_interface_scores[
                                                "binder_score"
                                            ],
                                            "Surface_Hydrophobicity": mpnn_interface_scores[
                                                "surface_hydrophobicity"
                                            ],
                                            "ShapeComplementarity": mpnn_interface_scores[
                                                "interface_sc"
                                            ],
                                            "PackStat": mpnn_interface_scores[
                                                "interface_packstat"
                                            ],
                                            "dG": mpnn_interface_scores[
                                                "interface_dG"
                                            ],
                                            "dSASA": mpnn_interface_scores[
                                                "interface_dSASA"
                                            ],
                                            "dG/dSASA": mpnn_interface_scores[
                                                "interface_dG_SASA_ratio"
                                            ],
                                            "Interface_SASA_%": mpnn_interface_scores[
                                                "interface_fraction"
                                            ],
                                            "Interface_Hydrophobicity": mpnn_interface_scores[
                                                "interface_hydrophobicity"
                                            ],
                                            "n_InterfaceResidues": mpnn_interface_scores[
                                                "interface_nres"
                                            ],
                                            "n_InterfaceHbonds": mpnn_interface_scores[
                                                "interface_interface_hbonds"
                                            ],
                                            "InterfaceHbondsPercentage": mpnn_interface_scores[
                                                "interface_hbond_percentage"
                                            ],
                                            "n_InterfaceUnsatHbonds": mpnn_interface_scores[
                                                "interface_delta_unsat_hbonds"
                                            ],
                                            "InterfaceUnsatHbondsPercentage": mpnn_interface_scores[
                                                "interface_delta_unsat_hbonds_percentage"
                                            ],
                                            "Site_RMSD": rmsd_site,
                                            "Target_RMSD": target_rmsd,
                                            "Alpha": mpnn_alpha,
                                            "Beta": mpnn_beta,
                                            "Loops": mpnn_loops,
                                            "Alpha_Interface": mpnn_alpha_interface,
                                            "Beta_Interface": mpnn_beta_interface,
                                            "Loops_Interface": mpnn_loops_interface,
                                            "Interface_AA": mpnn_interface_AA,
                                        }
                                    )

                            # predict binder alone
                            binder_statistics = predict_binder_alone(
                                binder_prediction_model,
                                mpnn_sequence["seq"],
                                mpnn_design_name,
                                design_paths,
                            )

                            # calculate average metrics
                            statistics_labels = [
                                "i_pLDDT",
                                "ss_pLDDT",
                                "Unrelaxed_Clashes",
                                "Relaxed_Clashes",
                                "Binder_Energy_Score",
                                "Surface_Hydrophobicity",
                                "ShapeComplementarity",
                                "PackStat",
                                "dG",
                                "dSASA",
                                "dG/dSASA",
                                "Interface_SASA_%",
                                "Interface_Hydrophobicity",
                                "n_InterfaceResidues",
                                "n_InterfaceHbonds",
                                "InterfaceHbondsPercentage",
                                "n_InterfaceUnsatHbonds",
                                "InterfaceUnsatHbondsPercentage",
                                "Site_RMSD",
                                "Target_RMSD",
                                "Alpha",
                                "Beta",
                                "Loops",
                                "Alpha_Interface",
                                "Beta_Interface",
                                "Loops_Interface",
                                "Interface_AA",
                            ]
                            mpnn_complex_averages, mpnn_complex_statistics = calculate_averages(
                                mpnn_complex_statistics, statistics_labels
                            )

                            binder_averages, binder_statistics = calculate_averages(
                                binder_statistics,
                                [
                                    "pLDDT",
                                    "pTM",
                                    "pAE",
                                    "Binder_RMSD",
                                ],
                            )

                            elapsed_mpnn_time = time.time() - mpnn_time
                            elapsed_mpnn_text = "%d hours, %d minutes, %d seconds" % (
                                int(elapsed_mpnn_time // 3600),
                                int((elapsed_mpnn_time % 3600) // 60),
                                int(elapsed_mpnn_time % 60),
                            )

                            seq_notes = validate_design_sequence(
                                mpnn_sequence["seq"],
                                mpnn_complex_averages.get("Relaxed_Clashes", 0),
                                advanced_settings,
                            )

                            # insert mpnn statistics
                            model_numbers = range(1, len(prediction_models) + 1)
                            mpnn_data = [
                                mpnn_design_name,
                                mpnn_sequence["seq"],
                                mpnn_score,
                                mpnn_seqid,
                            ]

                            for label in statistics_labels:
                                mpnn_data.append(
                                    mpnn_complex_averages.get(label, None)
                                )
                                for model in model_numbers:
                                    mpnn_data.append(
                                        mpnn_complex_statistics.get(model, {}).get(
                                            label, None
                                        )
                                    )

                            for label in [
                                "pLDDT",
                                "pTM",
                                "pAE",
                                "Binder_RMSD",
                            ]:
                                mpnn_data.append(binder_averages.get(label, None))
                                for model in model_numbers:
                                    mpnn_data.append(
                                        binder_statistics.get(model, {}).get(
                                            label, None
                                        )
                                    )

                            mpnn_data.extend(
                                [
                                    elapsed_mpnn_text,
                                    seq_notes,
                                    settings_file,
                                    filters_file,
                                    advanced_file,
                                ]
                            )

                            insert_data(mpnn_csv, mpnn_data)

                            plddt_values = {
                                i: mpnn_data[i]
                                for i in range(11, 15)
                                if mpnn_data[i] is not None
                            }
                            highest_plddt_key = int(max(plddt_values, key=plddt_values.get))
                            best_model_number = highest_plddt_key - 10
                            best_model_pdb = os.path.join(
                                design_paths["MPNN/Relaxed"],
                                f"{mpnn_design_name}_model{best_model_number}.pdb",
                            )

                            filter_conditions = check_filters(
                                mpnn_data, design_labels, filters
                            )
                            if filter_conditions is True:
                                print(mpnn_design_name + " passed all filters")
                                accepted_mpnn += 1
                                accepted_designs += 1

                                shutil.copy(best_model_pdb, design_paths["Accepted"])

                                final_data = [""] + mpnn_data
                                insert_data(final_csv, final_data)

                                if advanced_settings["save_design_animations"]:
                                    accepted_animation = os.path.join(
                                        design_paths["Accepted/Animation"],
                                        f"{design_name}.html",
                                    )
                                    if not os.path.exists(accepted_animation):
                                        shutil.copy(
                                            os.path.join(
                                                design_paths["Trajectory/Animation"],
                                                f"{design_name}.html",
                                            ),
                                            accepted_animation,
                                        )

                                plot_files = os.listdir(
                                    design_paths["Trajectory/Plots"]
                                )
                                plots_to_copy = [
                                    f
                                    for f in plot_files
                                    if f.startswith(design_name) and f.endswith(".png")
                                ]
                                for accepted_plot in plots_to_copy:
                                    source_plot = os.path.join(
                                        design_paths["Trajectory/Plots"],
                                        accepted_plot,
                                    )
                                    target_plot = os.path.join(
                                        design_paths["Accepted/Plots"],
                                        accepted_plot,
                                    )
                                    if not os.path.exists(target_plot):
                                        shutil.copy(source_plot, target_plot)

                            else:
                                print(
                                    f"Unmet filter conditions for {mpnn_design_name}"
                                )
                                failure_df = pd.read_csv(failure_csv)
                                special_prefixes = (
                                    "Average_",
                                    "1_",
                                    "2_",
                                    "3_",
                                    "4_",
                                    "5_",
                                )
                                incremented_columns = set()

                                for column in filter_conditions:
                                    base_column = column
                                    for prefix in special_prefixes:
                                        if column.startswith(prefix):
                                            base_column = column.split("_", 1)[1]

                                    if base_column not in incremented_columns:
                                        failure_df[base_column] = (
                                            failure_df[base_column] + 1
                                        )
                                        incremented_columns.add(base_column)

                                failure_df.to_csv(failure_csv, index=False)
                                shutil.copy(
                                    best_model_pdb, design_paths["Rejected"]
                                )

                            mpnn_n += 1

                            if (
                                accepted_mpnn
                                >= advanced_settings["max_mpnn_sequences"]
                            ):
                                break

                    if accepted_mpnn >= 1:
                        print(
                            "Found "
                            + str(accepted_mpnn)
                            + " MPNN designs passing filters"
                        )
                        print("")
                    else:
                        print(
                            "No accepted MPNN designs found for this trajectory."
                        )
                        print("")

                else:
                    print(
                        "Duplicate MPNN designs sampled with different trajectory, skipping current trajectory optimisation"
                    )
                    print("")

                if advanced_settings["remove_unrelaxed_trajectory"]:
                    os.remove(trajectory_pdb)

                design_time = time.time() - design_start_time
                design_time_text = "%d hours, %d minutes, %d seconds" % (
                    int(design_time // 3600),
                    int((design_time % 3600) // 60),
                    int(design_time % 60),
                )
                print(
                    "Design and validation of trajectory "
                    + design_name
                    + " took: "
                    + design_time_text
                )

            if (
                trajectory_n >= advanced_settings["start_monitoring"]
                and advanced_settings["enable_rejection_check"]
            ):
                acceptance = accepted_designs / trajectory_n
                if not acceptance >= advanced_settings["acceptance_rate"]:
                    print(
                        "The ratio of successful designs is lower than defined acceptance rate! Consider changing your design settings!"
                    )
                    print("Script execution stopping...")
                    break

        trajectory_n += 1
        gc.collect()

    elapsed_time = time.time() - script_start_time
    elapsed_text = "%d hours, %d minutes, %d seconds" % (
        int(elapsed_time // 3600),
        int((elapsed_time % 3600) // 60),
        int(elapsed_time % 60),
    )
    print(
        "Finished all designs. Script execution for "
        + str(trajectory_n)
        + " trajectories took: "
        + elapsed_text
    )

