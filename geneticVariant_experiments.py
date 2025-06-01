import argparse
import numpy as np
import os
import pickle
from tqdm import tqdm

from utils.Mixture import NMixture, PUMixture
import utils.mixtureUtils as mixtureUtils
from utils.NestedGroupDistUnknownGroup import NestedGroupDistUnknownGroup
import utils.correct_metrics as correct_metrics
import utils.correct_metrics_unknowngroups as correct_metrics_unknowngroups




def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser('get_fairness_metrics')
    parser.add_argument("--output", type=str, default="variants_results",
                        help="Path to file where the metrics are to be stored")
    parser.add_argument("--dim", type=int, default=None,
                        help="Dimension of synthetic data")
    parser.add_argument("--comp", type=int, default=None,
                        help="Number of Gaussian components per mixture")
    parser.add_argument("--threads", type=int, default=None,
                        help="Number of threads for NN training. If none, all cores will be used")
    parser.add_argument("--thresh", type=str, choices=['fixed', 'adaptive'], default='fixed',
                        help="Use classifier threshold for classification fixed or set it to the "
                             "estimated class prior in labeled data." 
                             "For fixed, the thresholds for supporting (0.737), moderate (0.829), and strong (0.932)"
                             "will be used.")
    parser.add_argument("--bootstrap_num", type=int, default=0,
                        help="Index of bootstrap sample. If 0 (default) no bootstraping is done")
    parser.add_argument("--compare", type=int, default=None,
                        help="Group 1 index (one of 1,2,3,4). If None, all groups 1-4 "
                             "will be compared to Group 0 (the majority group)")
    args = parser.parse_args()

    return args


def sample_bootstrap(data, rng):
    """
    Get a bootstrap sample from input array data.
    Sample will be same size as input, sampled with replacement from rng
    :param data: input array
    :param rng: numpy random generator
    """
    size = data.shape[0]
    indices = rng.choice(range(size), size=size, replace=True)
    samples = data[indices]
    return samples, indices


def main():

    args = parse_args()
    # 0.737  # supporting
    # 0.829  # moderate
    # 0.932  # strong
    dimensions = [2, 8] if args.dim is None else [args.dim]
    components = [2, 4, 8] if args.comp is None else [args.comp]

    num_groups = 2
    classifier_threshold = args.thresh
    calibrated_thresholds = {'benign_supporting': 0.391,
                             'benign_moderate': 0.197,
                             'benign_strong': 0.010,
                             'even': 0.5,
                             'supporting': 0.737,
                             'moderate': 0.829,
                             'strong': 0.932
                             }

    # Option to bootstrap
    bootstrap_idx = args.bootstrap_num
    bootstrap = bootstrap_idx > 0  # flag for bootstrapping

    # to save results
    folder_name = args.output
    os.makedirs(folder_name, exist_ok=True)

    # load dataset
    data_path = '../variants/processed_variants.npz'
    dataset = np.load(data_path, allow_pickle=True)

    for dim in dimensions:
        for comp in components:
            print(f"Dimension: {dim}, Components: {comp}")

            estimated_measures_ng = {measure: {"g1_l": list(), "g2_l": list(), "g1_ul": list(), "g2_ul": list(),
                                               "corrected_l": [],
                                               "corrected_u_l": []} for measure in ['dp', 'eo', 'pe', 'ppv']}
            estimated_measures_ng["bias"] = {"aucpn_g1": list(), "aucpn_g2": list(),
                                             "aucpp_g1": list(), "aucpp_g2": list(),
                                             "aucnn_g1": list(), "aucnn_g2": list(),
                                             'll': list(),
                                             'dim': dim, 'k': comp, 'thresh': list()}

            outfile_name = f"d{dim}K{comp}_variants"

            if bootstrap:
                file_name = f'{outfile_name}_b{bootstrap_idx}.npz'
            else:
                file_name = f'{outfile_name}.npz'

            file_path = os.path.join(folder_name, file_name)
            print(f'Saving results to {file_path}')

            s_labeled = dataset['labeled'][:, :dim]
            labels = dataset['labels']

            unlabeled_groups = dataset['unlabeled_groups']

            if args.compare is None:
                select = np.logical_or(unlabeled_groups == 0, unlabeled_groups != 0)
            else:
                select = np.logical_or(unlabeled_groups == 0, unlabeled_groups == args.compare)

            s_ub = dataset['unlabeled'][select, :dim]
            groups_ub = np.logical_not(unlabeled_groups[select] == 0).astype(int)
            s1_ub = s_ub[groups_ub == 0, :]  # group 1 unlabeled
            s2_ub = s_ub[groups_ub == 1, :]  # group 2 unlabeled

            if bootstrap:
                rng = np.random.default_rng(seed=bootstrap_idx)
                # sample positive & negative labeled
                pos_l = np.where(labels == 1)[0]  # positive indices
                neg_l = np.where(labels == 0)[0]  # negative indices
                pos_l_bs, _ = sample_bootstrap(pos_l, rng)
                neg_l_bs, _ = sample_bootstrap(neg_l, rng)
                labels = np.concatenate([labels[pos_l_bs], labels[neg_l_bs]])
                s_labeled = np.concatenate([s_labeled[pos_l_bs], s_labeled[neg_l_bs]])

                # sample group-specific samples
                s1_ub_idx, _ = sample_bootstrap(np.where(groups_ub == 0)[0], rng) # group 1 unlabeled
                s2_ub_idx, _ = sample_bootstrap(np.where(groups_ub == 1)[0], rng) # group 2 unlabeled
                s_ub = np.concatenate([s_ub[s1_ub_idx, :], s_ub[s2_ub_idx, :]])
                groups_ub = np.concatenate([groups_ub[s1_ub_idx], groups_ub[s2_ub_idx]])

            s_pos_bias = s_labeled[dataset['labels'] == 1]
            s_neg_bias = s_labeled[dataset['labels'] == 0]

            alpha_l = s_pos_bias.shape[0] / s_labeled.shape[0]
            eta = [s1_ub.shape[0] / s_ub.shape[0], s2_ub.shape[0] / s_ub.shape[0]]

            # load pre-trained scores
            if classifier_threshold == 'fixed':
                thresh_names = calibrated_thresholds.keys()
                class_thresholds = calibrated_thresholds.values()
            else:
                class_thresholds = [alpha_l]
                thresh_names = 'adaptive'

            print('Estimating selection bias model parameters')
            Kfit = [comp, comp]
            nested_group_EM = NestedGroupDistUnknownGroup(
                x_unlabeled=s_ub, x_labeled=[s_neg_bias, s_pos_bias],
                unlabeled_groups=groups_ub,
                components=Kfit, num_classes=2, num_groups=2)
            nested_group_EM.estimate_params(max_steps=1000)
            print('Parameters estimated')

            alphas_1_est, alphas_2_est = nested_group_EM.alphas
            w_1_est = [nested_group_EM.w[c_label][0] for c_label in [0, 1]]  # group==0 for each class (label)
            wl_est = [nested_group_EM.w_labeled[c_label] for c_label in [0, 1]]
            w_2_est = [nested_group_EM.w[c_label][1] for c_label in [0, 1]]  # group==1 for each class (label)
            lls_1 = lls_2 = nested_group_EM.lls

            for thresh_name, class_threshold in tqdm(zip(thresh_names, class_thresholds), total=len(class_thresholds)):
                if 'benign' in thresh_name:
                    # flip comparison direction
                    y_all_bias = dataset['labeled_scores'] <= class_threshold
                    y_all_ub = dataset['unlabeled_scores'][select] <= class_threshold
                    # flip meaning of "positive"
                    neg_label = 1
                    pos_label = 0
                    alpha_l = 1 - alpha_l

                else:
                    y_all_bias = dataset['labeled_scores'] >= class_threshold
                    y_all_ub = dataset['unlabeled_scores'][select] >= class_threshold
                    neg_label = 0
                    pos_label = 1

                y_pos_bias = y_all_bias[dataset['labels'] == pos_label]
                y_neg_bias = y_all_bias[dataset['labels'] == neg_label]

                y_g_ub = {g: y_all_ub[groups_ub == g] for g in np.unique(groups_ub)}

                estimated_measures_ng["bias"]['ll'].append(lls_1[-1])
                estimated_measures_ng["bias"]['thresh'].append(class_threshold)

                # Redefining mixtures with estimated parameters.
                # Distributions with estimated parameters are used for all subsequent operations
                estimated_d_unbiased = {
                    # first group
                    0: {"pos": NMixture(nested_group_EM.mu[pos_label], nested_group_EM.sg[pos_label], np.array(w_1_est[pos_label])),
                        "neg": NMixture(nested_group_EM.mu[neg_label], nested_group_EM.sg[neg_label], np.array(w_1_est[neg_label]))},
                    # second group:
                    1: {"pos": NMixture(nested_group_EM.mu[pos_label], nested_group_EM.sg[pos_label], np.array(w_2_est[pos_label])),
                        "neg": NMixture(nested_group_EM.mu[neg_label], nested_group_EM.sg[neg_label], np.array(w_2_est[neg_label]))}}

                estimated_d_biased = {
                    "pos": NMixture(nested_group_EM.mu[pos_label], nested_group_EM.sg[pos_label], np.array(wl_est[pos_label])),
                    "neg": NMixture(nested_group_EM.mu[neg_label], nested_group_EM.sg[neg_label], np.array(wl_est[neg_label]))}

                # Redefining alpha with the estimated parameter for each group
                alpha_est = [alphas_1_est[pos_label], alphas_2_est[pos_label]]  # todo: change this for > 2 groups

                # estimated biases
                # for a more stable AUC computation, we average 5 estimations of AUC
                # This is slow and we're not using AUC, so we skip it
                # num_auc_runs = 5
                # for g in range(num_groups):
                #     estimated_measures_ng["bias"][f"aucpp_g{g + 1}"].append(
                #         np.mean([mixtureUtils.auc(estimated_d_unbiased[g]["pos"], estimated_d_biased["pos"])
                #                  for _ in range(num_auc_runs)]))
                #     estimated_measures_ng["bias"][f"aucnn_g{g + 1}"].append(
                #         np.mean([mixtureUtils.auc(estimated_d_unbiased[g]["neg"], estimated_d_biased["neg"])
                #                  for _ in range(num_auc_runs)]))
                #     estimated_measures_ng["bias"][f"aucpn_g{g + 1}"].append(
                #         np.mean([mixtureUtils.auc(estimated_d_unbiased[g]["pos"], estimated_d_unbiased[g]["neg"])
                #                  for _ in range(num_auc_runs)]))

                # Unbiased group distribution
                estimated_p = {g: PUMixture(estimated_d_unbiased[g]['pos'], estimated_d_unbiased[g]['neg'], alpha_est[g])
                               for g in range(num_groups)}

                # Biased group distribution
                estimated_q = PUMixture(estimated_d_biased['pos'], estimated_d_biased['neg'], alpha_l)

                # Create a distribution object for the entire unlabeled set (not conditioned on a group)
                est_alpha_u = {pos_label: sum(eta[g] * alpha_est[g] for g in range(num_groups)),
                               neg_label: sum(eta[g] * (1 - alpha_est[g]) for g in range(num_groups))}
                est_wPos_u = sum(eta[g] * alpha_est[g] * estimated_d_unbiased[g]['pos'].ps for g in
                                 range(num_groups)) / est_alpha_u[pos_label]
                est_wNeg_u = sum(eta[g] * (1 - alpha_est[g]) * estimated_d_unbiased[g]['neg'].ps for g in
                                 range(num_groups)) / est_alpha_u[neg_label]
                est_d_pos_ub = NMixture(nested_group_EM.mu[pos_label], nested_group_EM.sg[pos_label], est_wPos_u)
                est_d_neg_ub = NMixture(nested_group_EM.mu[neg_label], nested_group_EM.sg[neg_label], est_wNeg_u)
                est_d_ub = PUMixture(est_d_pos_ub, est_d_neg_ub, est_alpha_u[pos_label])

                est_wPos_l = estimated_d_biased['pos'].ps
                est_wNeg_l = estimated_d_biased['neg'].ps

                est_d_pos_bias = estimated_d_biased['pos']
                est_d_neg_bias = estimated_d_biased['neg']

                YSoft_u = est_d_ub.pn_posterior(s_ub).reshape(-1, 1)
                # Since we do not have labels from the unlabeled data,
                # we use the posterior probability from the GMM as soft labels
                Y = np.concatenate([labels.reshape(-1, 1), YSoft_u])
                YPred = np.concatenate([y_all_bias, y_all_ub])
                WPos_l = {
                    g: estimated_d_unbiased[g]["pos"].points_pdf(s_labeled) / est_d_pos_bias.points_pdf(s_labeled)
                    for g in range(num_groups)}
                WPos_u = {g: estimated_d_unbiased[g]["pos"].points_pdf(s_ub) / est_d_pos_ub.points_pdf(s_ub)
                          for g in range(num_groups)}
                WNeg_l = {g: estimated_d_unbiased[g]["neg"].points_pdf(s_labeled) / est_d_neg_bias.points_pdf(
                    s_labeled) for g in range(num_groups)}
                WNeg_u = {g: estimated_d_unbiased[g]["neg"].points_pdf(s_ub) / est_d_neg_ub.points_pdf(s_ub)
                          for g in range(num_groups)}

                TPR = {g: correct_metrics.TPR(Y, YPred, class_threshold, np.concatenate((WPos_l[g], WPos_u[g])))
                       for g in range(num_groups)}
                FPR = {g: correct_metrics.FPR(Y, YPred, class_threshold, np.concatenate((WNeg_l[g], WNeg_u[g])))
                       for g in range(num_groups)}

                dp_group_difference, dp = correct_metrics_unknowngroups.correct_dp(y_all_bias=y_all_bias,
                                                                                   y_all_ub=y_g_ub,
                                                                                   s_all_bias=s_labeled,
                                                                                   p=estimated_p, q=estimated_q
                                                                                  )
                dp_bias_withUL = {g: TPR[g] + FPR[g] for g in range(num_groups)}
                estimated_measures_ng["dp"]['g1_l'].append(dp['dp_bias_withLabelled'][0])
                estimated_measures_ng["dp"]['g2_l'].append(dp['dp_bias_withLabelled'][1])
                estimated_measures_ng["dp"]['g1_ul'].append(dp_bias_withUL[0])
                estimated_measures_ng["dp"]['g2_ul'].append(dp_bias_withUL[1])
                estimated_measures_ng["dp"]["corrected_l"].append(dp_group_difference['corrected_l'])
                estimated_measures_ng["dp"]["corrected_u_l"].append((dp_bias_withUL[0] - dp_bias_withUL[1]))

                ## Equal opportunity
                eo_group_difference, eo = correct_metrics_unknowngroups.correct_eo(
                    y_pos_bias=y_pos_bias,
                    y_pos_ub={0: None, 1: None},  # this is ignored since we don't know the labels
                    s_pos_bias=s_pos_bias if pos_label==1 else s_neg_bias,
                    q_pos=estimated_d_biased["pos"],
                    p_pos={0: estimated_d_unbiased[0]["pos"], 1: estimated_d_unbiased[1]["pos"]})

                eo_bias_withUL = {g: TPR[g] for g in range(num_groups)}
                estimated_measures_ng["eo"]['g1_l'].append(eo['eo_bias_withLabelled'][0])
                estimated_measures_ng["eo"]['g2_l'].append(eo['eo_bias_withLabelled'][1])
                estimated_measures_ng["eo"]['g1_ul'].append(eo_bias_withUL[0])
                estimated_measures_ng["eo"]['g2_ul'].append(eo_bias_withUL[1])
                estimated_measures_ng["eo"]["corrected_l"].append(eo_group_difference['corrected_l'])
                estimated_measures_ng["eo"]["corrected_u_l"].append(eo_bias_withUL[0] - eo_bias_withUL[1])

                ## predictive equality
                pe_group_difference, pe = correct_metrics_unknowngroups.correct_pe(
                    y_neg_bias=y_neg_bias,
                    y_neg_ub={0: None, 1: None},  # this is ignored since we don't know the labels
                    s_neg_bias=s_neg_bias if pos_label == 1 else s_pos_bias,
                    q_neg=estimated_d_biased["neg"],
                    p_neg={0: estimated_d_unbiased[0]["neg"], 1: estimated_d_unbiased[1]["neg"]})

                pe_bias_withUL = {g: FPR[g] for g in range(num_groups)}
                estimated_measures_ng["pe"]['g1_l'].append(pe['pe_bias_withLabelled'][0])
                estimated_measures_ng["pe"]['g2_l'].append(pe['pe_bias_withLabelled'][1])
                estimated_measures_ng["pe"]['g1_ul'].append(pe_bias_withUL[0])
                estimated_measures_ng["pe"]['g2_ul'].append(pe_bias_withUL[1])
                estimated_measures_ng["pe"]["corrected_l"].append(pe_group_difference['corrected_l'])
                estimated_measures_ng["pe"]["corrected_u_l"].append(pe_bias_withUL[0] - pe_bias_withUL[1])

                # predictive positive value = TPR*alpha / (TPR*alpha + FPR*(1-alpha))
                ppv_group_difference = correct_metrics_unknowngroups.correct_ppv(
                    eo, pe, alpha_est, eo_bias_withUL, pe_bias_withUL)

                ppv_bias_withLabelled = {g: eo['eo_bias_withLabelled'][g] * alpha_est[g] / (
                        eo['eo_bias_withLabelled'][g] * alpha_est[g] + pe['pe_bias_withLabelled'][g] * (1 - alpha_est[g]))
                                         for g in range(num_groups)}
                ppv_bias_withUL = {g: eo_bias_withUL[g] * alpha_est[g] / (
                        eo_bias_withUL[g] * alpha_est[g] + pe_bias_withUL[g] * (1 - alpha_est[g]))
                                   for g in range(num_groups)}

                estimated_measures_ng["ppv"]['g1_l'].append(ppv_bias_withLabelled[0])
                estimated_measures_ng["ppv"]['g2_l'].append(ppv_bias_withLabelled[1])
                estimated_measures_ng["ppv"]['g1_ul'].append(ppv_bias_withUL[0])
                estimated_measures_ng["ppv"]['g2_ul'].append(ppv_bias_withUL[1])
                estimated_measures_ng["ppv"]["corrected_l"].append(ppv_group_difference['corrected_l'])
                estimated_measures_ng["ppv"]["corrected_u_l"].append(ppv_group_difference['corrected_u_l'])

            # Save fairness metrics data as npz files for each dimension
            np.savez(file_path, **{ 'estimated_ng': estimated_measures_ng})

            # Save estimated GMM model to file
            if not bootstrap:
                with open(file_path.split('.npz')[0]+'_gmm_model.pkl', 'wb') as f:
                    pickle.dump(nested_group_EM, f)


if __name__ == '__main__':
    main()