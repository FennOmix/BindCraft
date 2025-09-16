import os, shutil, math, pickle
import matplotlib.pyplot as plt
import numpy as np
import jax
import jax.numpy as jnp
from colabdesign import mk_afdesign_model, clear_mem
from colabdesign.af.alphafold.common import residue_constants
from colabdesign.af.loss import get_ptm, mask_loss, get_dgram_bins, _get_con_loss
from .biopython_utils import hotspot_residues, calculate_clash_score, calc_ss_percentage
from .generic_utils import update_failures

# hallucinate a binder

def binder_hallucination(design_name, starting_pdb, chain, target_hotspot_residues, length, seed,
                         helicity_value, design_models, advanced_settings, design_paths, failure_csv):
    model_pdb_path = os.path.join(design_paths["Trajectory"], design_name + ".pdb")

    # clear GPU memory for new trajectory
    clear_mem()

    # initialise binder hallucination model
    af_model = mk_afdesign_model(protocol="binder", debug=False, data_dir=advanced_settings["af_params_dir"],
                                use_multimer=advanced_settings["use_multimer_design"],
                                num_recycles=advanced_settings["num_recycles_design"],
                                best_metric='loss')

    # sanity check for hotspots
    if target_hotspot_residues == "":
        target_hotspot_residues = None

    af_model.prep_inputs(pdb_filename=starting_pdb, chain=chain, binder_len=length,
                         hotspot=target_hotspot_residues, seed=seed, rm_aa=advanced_settings["omit_AAs"],
                         rm_target_seq=advanced_settings["rm_template_seq_design"],
                         rm_target_sc=advanced_settings["rm_template_sc_design"])

    ### Update weights based on specified settings
    af_model.opt["weights"].update({
        "pae": advanced_settings["weights_pae_intra"],
        "plddt": advanced_settings["weights_plddt"],
        "i_pae": advanced_settings["weights_pae_inter"],
        "con": advanced_settings["weights_con_intra"],
        "i_con": advanced_settings["weights_con_inter"],
    })

    # redefine intramolecular contacts (con) and intermolecular contacts (i_con) definitions
    af_model.opt["con"].update({"num": advanced_settings["intra_contact_number"],
                                 "cutoff": advanced_settings["intra_contact_distance"],
                                 "binary": False,
                                 "seqsep": 9})
    af_model.opt["i_con"].update({"num": advanced_settings["inter_contact_number"],
                                   "cutoff": advanced_settings["inter_contact_distance"],
                                   "binary": False})

    ### additional loss functions
    if advanced_settings["use_rg_loss"]:
        add_rg_loss(af_model, advanced_settings["weights_rg"])

    if advanced_settings["use_i_ptm_loss"]:
        add_i_ptm_loss(af_model, advanced_settings["weights_iptm"])

    if advanced_settings["use_termini_distance_loss"]:
        add_termini_distance_loss(af_model, advanced_settings["weights_termini_loss"])

    # add the helicity loss
    add_helix_loss(af_model, helicity_value)

    # calculate the number of mutations to do based on the length of the protein
    greedy_tries = math.ceil(length * (advanced_settings["greedy_percentage"] / 100))

    ### start design algorithm based on selection
    if advanced_settings["design_algorithm"] == '2stage':
        af_model.design_pssm_semigreedy(soft_iters=advanced_settings["soft_iterations"],
                                        hard_iters=advanced_settings["greedy_iterations"],
                                        tries=greedy_tries, models=design_models,
                                        num_models=1, sample_models=advanced_settings["sample_models"],
                                        ramp_models=False, save_best=True)

    elif advanced_settings["design_algorithm"] == '3stage':
        af_model.design_3stage(soft_iters=advanced_settings["soft_iterations"],
                                temp_iters=advanced_settings["temporary_iterations"],
                                hard_iters=advanced_settings["hard_iterations"],
                                num_models=1, models=design_models,
                                sample_models=advanced_settings["sample_models"], save_best=True)

    elif advanced_settings["design_algorithm"] == 'greedy':
        af_model.design_semigreedy(advanced_settings["greedy_iterations"], tries=greedy_tries,
                                   num_models=1, models=design_models,
                                   sample_models=advanced_settings["sample_models"], save_best=True)

    elif advanced_settings["design_algorithm"] == 'mcmc':
        half_life = round(advanced_settings["greedy_iterations"] / 5, 0)
        t_mcmc = 0.01
        af_model._design_mcmc(advanced_settings["greedy_iterations"], half_life=half_life,
                              T_init=t_mcmc, mutation_rate=greedy_tries, num_models=1,
                              models=design_models, sample_models=advanced_settings["sample_models"],
                              save_best=True)

    elif advanced_settings["design_algorithm"] == '4stage':
        print("Stage 1: Test Logits")
        af_model.design_logits(iters=50, e_soft=0.9, models=design_models, num_models=1,
                               sample_models=advanced_settings["sample_models"], save_best=True)

        initial_plddt = get_best_plddt(af_model, length)

        if initial_plddt > 0.65:
            print("Initial trajectory pLDDT good, continuing: " + str(initial_plddt))
            if advanced_settings["optimise_beta"]:
                af_model.save_pdb(model_pdb_path)
                _, beta, *_ = calc_ss_percentage(model_pdb_path, advanced_settings, 'B')
                os.remove(model_pdb_path)

                if float(beta) > 15:
                    advanced_settings["soft_iterations"] += advanced_settings["optimise_beta_extra_soft"]
                    advanced_settings["temporary_iterations"] += advanced_settings["optimise_beta_extra_temp"]
                    af_model.set_opt(num_recycles=advanced_settings["optimise_beta_recycles_design"])
                    print("Beta sheeted trajectory detected, optimising settings")

            logits_iter = advanced_settings["soft_iterations"] - 50
            if logits_iter > 0:
                print("Stage 1: Additional Logits Optimisation")
                af_model.clear_best()
                af_model.design_logits(iters=logits_iter, e_soft=1, models=design_models,
                                       num_models=1, sample_models=advanced_settings["sample_models"],
                                       ramp_recycles=False, save_best=True)
                af_model._tmp["seq_logits"] = af_model.aux["seq"]["logits"]
                logit_plddt = get_best_plddt(af_model, length)
                print("Optimised logit trajectory pLDDT: " + str(logit_plddt))
            else:
                logit_plddt = initial_plddt

            if advanced_settings["temporary_iterations"] > 0:
                print("Stage 2: Softmax Optimisation")
                af_model.clear_best()
                af_model.design_soft(advanced_settings["temporary_iterations"], e_temp=1e-2,
                                     models=design_models, num_models=1,
                                     sample_models=advanced_settings["sample_models"],
                                     ramp_recycles=False, save_best=True)
                softmax_plddt = get_best_plddt(af_model, length)
            else:
                softmax_plddt = logit_plddt

            if softmax_plddt > 0.65:
                print("Softmax trajectory pLDDT good, continuing: " + str(softmax_plddt))
                if advanced_settings["hard_iterations"] > 0:
                    af_model.clear_best()
                    print("Stage 3: One-hot Optimisation")
                    af_model.design_hard(advanced_settings["hard_iterations"], temp=1e-2,
                                          models=design_models, num_models=1,
                                          sample_models=advanced_settings["sample_models"], dropout=False,
                                          ramp_recycles=False, save_best=True)
                    onehot_plddt = get_best_plddt(af_model, length)

                if onehot_plddt > 0.65:
                    print("One-hot trajectory pLDDT good, continuing: " + str(onehot_plddt))
                    if advanced_settings["greedy_iterations"] > 0:
                        print("Stage 4: PSSM Semigreedy Optimisation")
                        af_model.design_pssm_semigreedy(soft_iters=0,
                                                         hard_iters=advanced_settings["greedy_iterations"],
                                                         tries=greedy_tries, models=design_models,
                                                         num_models=1, sample_models=advanced_settings["sample_models"],
                                                         ramp_models=False, save_best=True)
                else:
                    update_failures(failure_csv, 'Trajectory_one-hot_pLDDT')
                    print("One-hot trajectory pLDDT too low to continue: " + str(onehot_plddt))
            else:
                update_failures(failure_csv, 'Trajectory_softmax_pLDDT')
                print("Softmax trajectory pLDDT too low to continue: " + str(softmax_plddt))
        else:
            update_failures(failure_csv, 'Trajectory_logits_pLDDT')
            print("Initial trajectory pLDDT too low to continue: " + str(initial_plddt))
    else:
        print("ERROR: No valid design model selected")
        exit()
        return

    final_plddt = get_best_plddt(af_model, length)
    af_model.save_pdb(model_pdb_path)
    af_model.aux["log"]["terminate"] = ""

    ca_clashes = calculate_clash_score(model_pdb_path, 2.5, only_ca=True)

    if ca_clashes > 0:
        af_model.aux["log"]["terminate"] = "Clashing"
        update_failures(failure_csv, 'Trajectory_Clashes')
        print("Severe clashes detected, skipping analysis and MPNN optimisation")
        print("")
    else:
        if final_plddt < 0.7:
            af_model.aux["log"]["terminate"] = "LowConfidence"
            update_failures(failure_csv, 'Trajectory_final_pLDDT')
            print("Trajectory starting confidence low, skipping analysis and MPNN optimisation")
            print("")
        else:
            binder_contacts = hotspot_residues(model_pdb_path)
            binder_contacts_n = len(binder_contacts.items())

            if binder_contacts_n < 3:
                af_model.aux["log"]["terminate"] = "LowConfidence"
                update_failures(failure_csv, 'Trajectory_Contacts')
                print("Too few contacts at the interface, skipping analysis and MPNN optimisation")
                print("")
            else:
                af_model.aux["log"]["terminate"] = ""
                print("Trajectory successful, final pLDDT: " + str(final_plddt))

    if af_model.aux["log"]["terminate"] != "":
        shutil.move(model_pdb_path, design_paths[f"Trajectory/{af_model.aux['log']['terminate']}"])

    af_model.get_seqs()
    if advanced_settings["save_design_trajectory_plots"]:
        plot_trajectory(af_model, design_name, design_paths)

    if advanced_settings["save_design_animations"]:
        plots = af_model.animate(dpi=150)
        with open(os.path.join(design_paths["Trajectory/Animation"], design_name + ".html"), 'w') as f:
            f.write(plots)
        plt.close('all')

    if advanced_settings["save_trajectory_pickle"]:
        with open(os.path.join(design_paths["Trajectory/Pickle"], design_name + ".pickle"), 'wb') as handle:
            pickle.dump(af_model.aux['all'], handle, protocol=pickle.HIGHEST_PROTOCOL)

    return af_model


def get_best_plddt(af_model, length):
    return round(np.mean(af_model._tmp["best"]["aux"]["plddt"][-length:]), 2)


def add_rg_loss(self, weight=0.1):
    """add radius of gyration loss"""
    def loss_fn(inputs, outputs):
        xyz = outputs["structure_module"]
        ca = xyz["final_atom_positions"][:, residue_constants.atom_order["CA"]]
        ca = ca[-self._binder_len:]
        rg = jnp.sqrt(jnp.square(ca - ca.mean(0)).sum(-1).mean() + 1e-8)
        rg_th = 2.38 * ca.shape[0] ** 0.365
        rg = jax.nn.elu(rg - rg_th)
        return {"rg": rg}

    self._callbacks["model"]["loss"].append(loss_fn)
    self.opt["weights"]["rg"] = weight


def add_i_ptm_loss(self, weight=0.1):
    def loss_iptm(inputs, outputs):
        p = 1 - get_ptm(inputs, outputs, interface=True)
        i_ptm = mask_loss(p)
        return {"i_ptm": i_ptm}

    self._callbacks["model"]["loss"].append(loss_iptm)
    self.opt["weights"]["i_ptm"] = weight


def add_helix_loss(self, weight=0):
    def binder_helicity(inputs, outputs):
        if "offset" in inputs:
            offset = inputs["offset"]
        else:
            idx = inputs["residue_index"].flatten()
            offset = idx[:, None] - idx[None, :]

        dgram = outputs["distogram"]["logits"]
        dgram_bins = get_dgram_bins(outputs)
        mask_2d = np.outer(np.append(np.zeros(self._target_len), np.ones(self._binder_len)),
                           np.append(np.zeros(self._target_len), np.ones(self._binder_len)))

        x = _get_con_loss(dgram, dgram_bins, cutoff=6.0, binary=True)
        if offset is None:
            if mask_2d is None:
                helix_loss = jnp.diagonal(x, 3).mean()
            else:
                helix_loss = jnp.diagonal(x * mask_2d, 3).sum() + (jnp.diagonal(mask_2d, 3).sum() + 1e-8)
        else:
            mask = offset == 3
            if mask_2d is not None:
                mask = jnp.where(mask_2d, mask, 0)
            helix_loss = jnp.where(mask, x, 0.0).sum() / (mask.sum() + 1e-8)

        return {"helix": helix_loss}

    self._callbacks["model"]["loss"].append(binder_helicity)
    self.opt["weights"]["helix"] = weight


def add_termini_distance_loss(self, weight=0.1, threshold_distance=7.0):
    """Add loss penalizing the distance between N and C termini"""
    def loss_fn(inputs, outputs):
        xyz = outputs["structure_module"]
        ca = xyz["final_atom_positions"][:, residue_constants.atom_order["CA"]]
        ca = ca[-self._binder_len:]

        n_terminus = ca[0]
        c_terminus = ca[-1]

        termini_distance = jnp.linalg.norm(n_terminus - c_terminus)
        deviation = jax.nn.elu(termini_distance - threshold_distance)
        termini_distance_loss = jax.nn.relu(deviation)
        return {"NC": termini_distance_loss}

    self._callbacks["model"]["loss"].append(loss_fn)
    self.opt["weights"]["NC"] = weight


def plot_trajectory(af_model, design_name, design_paths):
    metrics_to_plot = ['loss', 'plddt', 'ptm', 'i_ptm', 'con', 'i_con', 'pae', 'i_pae', 'rg', 'mpnn']
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

    for index, metric in enumerate(metrics_to_plot):
        if metric in af_model.aux["log"]:
            plt.figure()
            loss = af_model.get_loss(metric)
            iterations = range(1, len(loss) + 1)
            plt.plot(iterations, loss, label=f'{metric}', color=colors[index % len(colors)])
            plt.xlabel('Iterations')
            plt.ylabel(metric)
            plt.title(design_name)
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(design_paths["Trajectory/Plots"], f"{design_name}_{metric}.png"), dpi=150)
            plt.close()
