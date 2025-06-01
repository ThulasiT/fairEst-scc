import argparse
import numpy as np
import os
import pickle
import pandas as pd
from tqdm import tqdm

from pnuem.generate import generateSamples
from pnuem.Mixture import NMixture, PUMixture
import pnuem.mixtureUtils as mixtureUtils
from pnuem.NestedGroupDist import NestedGroupDist
from pnuem.NestedGroupDistUnknownGroup import NestedGroupDistUnknownGroup
import pnuem.correct_metrics as correct_metrics
import pnuem.correct_metrics_unknowngroups as correct_metrics_unknowngroups

parser = argparse.ArgumentParser('get_fairness_metrics')
parser.add_argument("--output", type=str, default="synthetic_results",
                    help="Path to file where the metrics are to be stored")
parser.add_argument("--dataset_path", type=str, default="synthetic_data",
                    help="Path with saved datasets")
parser.add_argument("--savemodel", action='store_true',
                    help="Use this flag to save estimated model parameters to file. Will be saved to output path set by --output")
parser.add_argument("--etaType", type=str, choices=["equal", "minority"], nargs="?", default="equal",
                    help="Choose whether the groups are equally populated or minority")
parser.add_argument("--estimate", action='store_true',
                    help="Flag to estimate parameters. If pasded, GMM will be used to estimate distributions")
parser.add_argument("--estimateNG", action='store_true',
                    help="Flag to estimate parameters. If pasded, GMM will be used to estimate distributions")
parser.add_argument("--dim", type=int, default=None,
                    help="Dimension of synthetic data")
parser.add_argument("--lam", type=float, default=0.5,
                    help="Proportion of datapoints that have labels. Default 0.5")
parser.add_argument("--comp", type=int, default=None,
                    help="Number of Gaussian components per mixture")
parser.add_argument("--threads", type=int, default=None,
                    help="Number of threads for NN training. If none, all cores will be used")
parser.add_argument("--auc", type=int, choices=[0, 1, 2, 10], default=10,
                    help="AUC range, 0: [0.65, 0.75], 1:[0.75, 0.85], 2:[0.85, 0.95], 10: run all three ranges")
args = parser.parse_args()

# Set eta value in accordance with etaType
if args.etaType == "equal":
    # Both groups of interest are similarly populated
    eta = [0.5, 0.5]  # eta= [a,b] In unbiased data, a% is G1 and b% is G2  #adds up to 1
    eta_ = [0.5, 0.5]  # eta_= [a,b] In biased data, a% is G1 and b% is G2  #adds up to 1
else:
    # One group has much lesser data points than the other. This is further exaggerated in biased data
    eta = [0.1, 0.9]
    eta_ = [0.05, 0.95]

aucpn_ranges = [[0.65, 0.75], [0.75, 0.85], [0.85, 0.95]]
if args.auc == 10:  # run all AUC ranges
    aucpn_ranges = aucpn_ranges
    auc_idxs = [0, 1, 2]
else:
    aucpn_ranges = [aucpn_ranges[args.auc]]
    auc_idxs = [args.auc]

dimensions = [2, 8] if args.dim is None else [args.dim]
components = [2, 4, 8] if args.comp is None else [args.comp]

total_samples = 25000  # Total number of data samples
lam = args.lam  # Proportion of biased (labeled) points in the total data

num_groups = 2
classifier_threshold = 'fixed' # classifier threshold set to 0.5. Use 'alternative' to use estimated alpha

# Fixed alphas
alpha = [0.5, 0.5]  # Fraction of positive unbiased samples from each group
# 1 - alpha fraction of negative unbiased samples from each group

alpha_ = [0.5, 0.5]  # Fraction of positive biased samples from each group
# 1 - alpha_ negative biased samples from each group

# to save results
folder_name = args.output
os.makedirs(folder_name, exist_ok=True)
# data path
dataset_path = args.dataset_path

# to save estimated model parameters to pickle file
save_model_params = args.savemodel

for dim in dimensions:
    for comp in components:
        for aucpn, aucpn_range in zip(auc_idxs, aucpn_ranges):

            oracle_measures = {measure: {"uncorrected": list(),
                                         "corrected": list(),
                                         "corrected_l": list(),
                                         "corrected_u_l": list()} for measure in ['dp', 'eo', 'pe', 'ppv']}
            oracle_measures["bias"] = {"aucpn_g1": list(), "aucpn_g2": list(),
                                       "aucpp_g1": list(), "aucpp_g2": list(),
                                       "aucnn_g1": list(), "aucnn_g2": list(),
                                       "avg_sb": list(),
                                       "alpha": alpha, 'alpha_l': alpha_, 'lambda': lam}  # save setting
            estimated_measures = {measure: {"uncorrected": [],
                                            "corrected": [],
                                            "corrected_l": [],
                                            "corrected_u_l": []} for measure in ['dp', 'eo', 'pe', 'ppv']}
            estimated_measures["bias"] = {"aucpn_g1": list(), "aucpn_g2": list(),
                                          "aucpp_g1": list(), "aucpp_g2": list(),
                                          "aucnn_g1": list(), "aucnn_g2": list()}

            estimated_measures_ng = {measure: {"corrected_l": [],
                                               "corrected_u_l": []} for measure in ['dp', 'eo', 'pe', 'ppv']}
            estimated_measures_ng["bias"] = {"aucpn_g1": list(), "aucpn_g2": list(),
                                             "aucpp_g1": list(), "aucpp_g2": list(),
                                             "aucnn_g1": list(), "aucnn_g2": list()}

            print(f"Dimension: {dim}, Components: {comp}, AUC range: {aucpn_range}")
            # To save results

            file_name = f"d{dim}K{comp}auc{aucpn_range[0]}_{lam}_{args.etaType}Eta.npz"
            file_path = os.path.join(folder_name, file_name)
            print(f'Saving results to {file_path}')

            pkl_file_name = os.path.join(dataset_path, 'parameters', f'synthetic_datasets_params_d{dim}K{comp}.pkl')
            params = pd.read_pickle(pkl_file_name)

            # Create a dataframe with these AUCs
            df = params[(params['aucpn'] > aucpn_range[0]) & (params['aucpn'] < aucpn_range[1])]

            for i in tqdm(range(df.shape[0])):

                # Positive distribution
                mu = {1: df['mu_p'].iloc[i],  # positive label
                      0: df['mu_n'].iloc[i]}  # negative label
                sig = {1: (df['sig_p'].iloc[i]),
                       0: (df['sig_n'].iloc[i])}
                component_weights = {0:  # group 0
                                         {1: df['w_p'].iloc[i],
                                          0: df['w_n'].iloc[i]},
                                     1:  # group 1
                                         {1: df['w_p_g2'].iloc[i],
                                          0: df['w_n_g2'].iloc[i]}}
                component_weights_labeled = {0:
                                                 {1: df['wl_p'].iloc[i],
                                                  0: df['wl_n'].iloc[i]},
                                             1:
                                                 {1: df['wl_p_g2'].iloc[i],
                                                  0: df['wl_n_g2'].iloc[i]}}

                distrib_ub = {g: {label: NMixture(mu[label], sig[label], np.array(component_weights[g][label]))
                                  for label in [1, 0]} for g in range(num_groups)}

                distrib_bias = {
                    g: {label: NMixture(mu[label], sig[label], np.array(component_weights_labeled[g][label]))
                        for label in [1, 0]} for g in range(num_groups)}

                # Oracle distributions
                p_oracle = {g: PUMixture(distrib_ub[g][1], distrib_ub[g][0], alpha[g]) for g in range(num_groups)}

                # Biased group distribution
                q_oracle = {g: PUMixture(distrib_bias[g][1], distrib_bias[g][0], alpha_[g]) for g in range(num_groups)}

                # Create a distribution object for the entire unlabeled set (not conditioned on a group)
                alpha_u = {1: sum(eta[g] * alpha[g] for g in range(num_groups)),
                           0: sum(eta[g] * (1 - alpha[g]) for g in range(num_groups))}

                wPos_u = sum(eta[g] * alpha[g] * component_weights[g][1] for g in range(num_groups)) / alpha_u[1]
                wNeg_u = sum(eta[g] * (1 - alpha[g]) * component_weights[g][0] for g in range(num_groups)) / alpha_u[0]

                d_pos_ub = NMixture(mu[1], sig[1], wPos_u)
                d_neg_ub = NMixture(mu[0], sig[0], wNeg_u)
                d_ub = PUMixture(d_pos_ub, d_neg_ub, alpha_u[1])

                # Create a distribution object for the entire labeled set (not conditioned on a group)
                # wPos_l and wNeg_l can be estimated directly from a different fromulation of the GMM estimation, where the labeled data is treated as one sample
                # instead of two samples (one for each group).
                alpha_l = {1: sum(eta_[g] * alpha_[g] for g in range(num_groups)),
                           0: sum(eta_[g] * (1 - alpha_[g]) for g in range(num_groups))}
                wPos_l = sum(eta_[g] * alpha_[g] * component_weights_labeled[g][1] for g in range(num_groups)) / alpha_l[1]
                wNeg_l = sum(
                    eta_[g] * (1 - alpha_[g]) * component_weights_labeled[g][0] for g in range(num_groups)) / alpha_l[0]

                d_pos_bias = NMixture(mu[1], sig[1], wPos_l)
                d_neg_bias = NMixture(mu[0], sig[0], wNeg_l)
                d_bias = PUMixture(d_pos_bias, d_neg_bias, alpha_l[1])

                # load existing data
                loaded_data = pd.read_csv(os.path.join(dataset_path,'samples', f'synthd{dim}K{comp}A{aucpn}_{lam}_{eta[0]}_{i}_numerical.csv'), index_col=0)
                loaded_scores = pd.read_csv(os.path.join(dataset_path, 'samples', f'synthd{dim}K{comp}A{aucpn}_{lam}_{eta[0]}_{i}_group_scores_remapped.csv'), index_col=0)
                loaded_data = loaded_data.merge(loaded_scores , left_index=True, right_index=True) # merge on index
                assert (np.all(loaded_data['group_x']==loaded_data['group_y'])) & np.all(loaded_data['label_x']==loaded_data['label_y'])
                loaded_data.drop(columns=['group_y', 'label_y'], inplace=True)
                loaded_data.rename(columns={'group_x':'group', 'label_x':'label'}, inplace=True)


                cols = [str(d_) for d_ in range(dim)]
                s1_pos_ub = loaded_data[(loaded_data.group==0) & (loaded_data.label==1) & ~(loaded_data.bias)][cols].values
                s1_neg_ub = loaded_data[(loaded_data.group==0) & (loaded_data.label==0) & ~(loaded_data.bias)][cols].values
                s2_pos_ub = loaded_data[(loaded_data.group==1) & (loaded_data.label==1) & ~(loaded_data.bias)][cols].values
                s2_neg_ub = loaded_data[(loaded_data.group==1) & (loaded_data.label==0) & ~(loaded_data.bias)][cols].values
                s1_pos_bias = loaded_data[(loaded_data.group==0) & (loaded_data.label==1) & (loaded_data.bias)][cols].values
                s1_neg_bias = loaded_data[(loaded_data.group==0) & (loaded_data.label==0) & (loaded_data.bias)][cols].values
                s2_pos_bias = loaded_data[(loaded_data.group==1) & (loaded_data.label==1) & (loaded_data.bias)][cols].values
                s2_neg_bias = loaded_data[(loaded_data.group==1) & (loaded_data.label==0) & (loaded_data.bias)][cols].values

                
                # All unlabeled samples
                s_ub = np.concatenate((s1_pos_ub, s1_neg_ub, s2_pos_ub, s2_neg_ub), axis=0)
                s_pos_ub = np.concatenate([s1_pos_ub, s2_pos_ub])
                s_neg_ub = np.concatenate([s1_neg_ub, s2_neg_ub])
                s1_ub = np.concatenate((s1_pos_ub, s1_neg_ub), axis=0)
                s2_ub = np.concatenate((s2_pos_ub, s2_neg_ub), axis=0)

                # Positive and negative labeled samples
                s_pos_bias = np.concatenate((s1_pos_bias, s2_pos_bias), axis=0)
                s_neg_bias = np.concatenate((s1_neg_bias, s2_neg_bias), axis=0)
                # labeled data for each group
                s1_bias = np.concatenate((s1_pos_bias, s1_neg_bias), axis=0)
                s2_bias = np.concatenate((s2_pos_bias, s2_neg_bias), axis=0)
                s_all_bias = np.concatenate([s1_bias, s2_bias])

                # for a more stable AUC computation, we average 5 estimations of AUC
                num_auc_runs = 5
                for g in range(num_groups):
                    oracle_measures["bias"][f"aucpp_g{g + 1}"].append(
                        np.mean([mixtureUtils.auc(p_oracle[g].pos, q_oracle[g].pos) for _ in range(num_auc_runs)]))
                    oracle_measures["bias"][f"aucnn_g{g + 1}"].append(
                        np.mean([mixtureUtils.auc(p_oracle[g].neg, q_oracle[g].neg) for _ in range(num_auc_runs)]))
                    oracle_measures["bias"][f"aucpn_g{g + 1}"].append(
                        np.mean([mixtureUtils.auc(p_oracle[g].pos, p_oracle[g].neg, alpha[g]) or _ in range(num_auc_runs)]))
                oracle_measures["bias"]['avg_sb'].append( (df['aucpp'].iloc[i]+
                                                          df['aucpp_g2'].iloc[i]+
                                                          df['aucpp'].iloc[i]+
                                                          df['aucpp_g2'].iloc[i])/4)
                # Training a 2-layer neural network classifier

                # Using the biased samples to train the model
                X = np.vstack((s_pos_bias, s_neg_bias))
                y = np.hstack((np.ones(len(s_pos_bias)), np.zeros(len(s_neg_bias))))

                # Get the trained model on this data
                #model = getModel(X, y, threads=args.threads)

                if classifier_threshold == 'fixed':
                    class_threshold = 0.5
                else:
                    class_threshold = alpha_l[1]

                # load model scores
                y_bias = {'pos': {g: loaded_data[(loaded_data.group==g) & (loaded_data.label==1) & (loaded_data.bias)]['score_MultiLayerPerceptron'].values
                                    for g in range(num_groups)},
                          'neg': {g: loaded_data[(loaded_data.group==g) & (loaded_data.label==0) & (loaded_data.bias)]['score_MultiLayerPerceptron'].values
                                    for g in range(num_groups)}}

                y_ub = {'pos': {g: loaded_data[(loaded_data.group==g) & (loaded_data.label==1) & ~(loaded_data.bias)]['score_MultiLayerPerceptron'].values
                                    for g in range(num_groups)},
                        'neg': {g: loaded_data[(loaded_data.group==g) & (loaded_data.label==0) & ~(loaded_data.bias)]['score_MultiLayerPerceptron'].values
                                    for g in range(num_groups)}} 
                
                # Bias group 1
                y1_all_bias = np.concatenate((y_bias['pos'][0], y_bias['neg'][0]), axis=0) >= class_threshold
                y1_pos_bias = y_bias['pos'][0] >= class_threshold
                y1_neg_bias = y_bias['neg'][0] >= class_threshold

                # Bias group 2
                y2_all_bias = np.concatenate((y_bias['pos'][1], y_bias['neg'][1]), axis=0) >= class_threshold
                y2_pos_bias = y_bias['pos'][1] >= class_threshold
                y2_neg_bias = y_bias['neg'][1] >= class_threshold
                y_all_bias = np.concatenate([y1_all_bias, y2_all_bias])

                # Unbiased group 1
                y1_all_ub = np.concatenate((y_ub['pos'][0], y_ub['neg'][0]), axis=0) >= class_threshold
                y1_pos_ub = y_ub['pos'][0] >= class_threshold
                y1_neg_ub = y_ub['neg'][0] >= class_threshold

                # Unbiased group 2
                y2_all_ub = np.concatenate((y_ub['pos'][1], y_ub['neg'][1]), axis=0) >= class_threshold
                y2_pos_ub = y_ub['pos'][1] >= class_threshold
                y2_neg_ub = y_ub['neg'][1] >= class_threshold
               
                y_pos_bias = np.concatenate([y1_pos_bias, y2_pos_bias])
                y_neg_bias = np.concatenate([y1_neg_bias, y2_neg_bias])
                y_all_ub = np.concatenate([y1_all_ub, y2_all_ub])
                y_pos_ub = np.concatenate([y1_pos_ub, y2_pos_ub])
                y_neg_ub = np.concatenate([y1_neg_ub, y2_neg_ub])

                label_all_bias = np.concatenate([np.ones(s1_pos_bias.shape[0]), np.zeros(s1_neg_bias.shape[0]), \
                                                 np.ones(s2_pos_bias.shape[0]), np.zeros(s2_neg_bias.shape[0])])
                label_all_bias = label_all_bias.reshape(-1, 1)

                # Compute TPR and FPR for groups for Correction with labelled and unlabelled points

                # Since we do not have labels from the unlabeled data,
                # we use the posterior probability from the GMM as soft labels
                YSoft_u = d_ub.pn_posterior(s_ub).reshape(-1, 1)
                Y = np.concatenate([label_all_bias, YSoft_u])
                YPred = np.concatenate([y_all_bias, y_all_ub])
                WPos_l = {g: distrib_ub[g][1].points_pdf(s_all_bias)/d_pos_bias.points_pdf(s_all_bias) for g in range(num_groups)}
                WPos_u = {g: distrib_ub[g][1].points_pdf(s_ub) / d_pos_ub.points_pdf(s_ub) for g in range(num_groups)}
                TPR = {g: correct_metrics.TPR(Y, YPred, class_threshold, np.concatenate((WPos_l[g], WPos_u[g]))) for g in range(num_groups)}

                WNeg_l = {g: distrib_ub[g][0].points_pdf(s_all_bias) / d_neg_bias.points_pdf(s_all_bias) for g in range(num_groups)}
                WNeg_u = {g: distrib_ub[0][0].points_pdf(s_ub) / d_neg_ub.points_pdf(s_ub) for g in range(num_groups)}
                FPR = {g: correct_metrics.FPR(Y, YPred, class_threshold, np.concatenate((WNeg_l[g], WNeg_u[g]))) for g in range(num_groups)}

                # Compute fairness metrics (Oracle)
                # Demographic parity
                dp_group_difference, _ = correct_metrics.correct_dp(points_bias={0: s1_bias, 1: s2_bias},
                                                                    y_all_bias={0: y1_all_bias, 1: y2_all_bias},
                                                                    y_all_ub={0: y1_all_ub, 1: y2_all_ub},
                                                                    p={g: p_oracle[g] for g in range(num_groups)},
                                                                    q={g: q_oracle[g] for g in range(num_groups)})

                dp_bias_withLabelled = {
                    g: sum(y_all_bias * p_oracle[g].points_pdf(s_all_bias) / d_bias.points_pdf(s_all_bias)) / y_all_bias.shape[0]
                    for g in range(num_groups)}
                dp_bias_withUL = {g: alpha[g] * TPR[g] + (1-alpha[g]) * FPR[g] for g in range(num_groups)}

                oracle_measures["dp"]["uncorrected"].append(
                    abs(dp_group_difference['uncorrected'] - dp_group_difference['unbiased']))
                oracle_measures["dp"]["corrected"].append(
                    abs(dp_group_difference['corrected'] - dp_group_difference['unbiased']))
                oracle_measures["dp"]["corrected_l"].append(
                    abs((dp_bias_withLabelled[0] - dp_bias_withLabelled[1]) - dp_group_difference['unbiased']))
                oracle_measures["dp"]["corrected_u_l"].append(
                    abs((dp_bias_withUL[0] - dp_bias_withUL[1]) - dp_group_difference['unbiased']))

                ## Equal opportunity
                eo_group_difference, eo = correct_metrics.correct_eo(points_pos_bias={0: s1_pos_bias, 1: s2_pos_bias},
                                                                     y_pos_bias={0: y1_pos_bias, 1: y2_pos_bias},
                                                                     y_pos_ub={0: y1_pos_ub, 1: y2_pos_ub},
                                                                     d_pos_bias={g: distrib_bias[g][1] for g in
                                                                                 range(num_groups)},
                                                                     d_pos_ub={g: distrib_ub[g][1] for g in
                                                                               range(num_groups)})

                eo_bias_withLabelled = {
                    g: sum(y_pos_bias * distrib_ub[g][1].points_pdf(s_pos_bias) / d_pos_bias.points_pdf(s_pos_bias)) /
                       y_pos_bias.shape[0]
                    for g in range(num_groups)}
                eo_bias_withUL = {g: TPR[g] for g in range(num_groups)}

                oracle_measures["eo"]["uncorrected"].append(
                    abs(eo_group_difference['uncorrected'] - eo_group_difference['unbiased']))
                oracle_measures["eo"]["corrected"].append(
                    abs(eo_group_difference['corrected'] - eo_group_difference['unbiased']))
                oracle_measures["eo"]["corrected_l"].append(
                    abs((eo_bias_withLabelled[0] - eo_bias_withLabelled[1]) - eo_group_difference['unbiased']))
                oracle_measures["eo"]["corrected_u_l"].append(
                    abs((eo_bias_withUL[0] - eo_bias_withUL[1]) - eo_group_difference['unbiased']))

                ## predictive equality
                pe_group_difference, pe = correct_metrics.correct_pe(points_neg_bias={0: s1_neg_bias, 1: s2_neg_bias},
                                                                     y_neg_bias={0: y1_neg_bias, 1: y2_neg_bias},
                                                                     y_neg_ub={0: y1_neg_ub, 1: y2_neg_ub},
                                                                     d_neg_bias={g: distrib_bias[g][0] for g in
                                                                                 range(num_groups)},
                                                                     d_neg_ub={g: distrib_ub[g][0] for g in
                                                                               range(num_groups)})

                pe_bias_withLabelled = {
                    g: sum(y_neg_bias * distrib_ub[g][0].points_pdf(s_neg_bias) / d_neg_bias.points_pdf(s_neg_bias)) /
                       y_neg_bias.shape[0]
                    for g in range(num_groups)}
                pe_bias_withUL = {g: FPR[g] for g in range(num_groups)}

                oracle_measures["pe"]["uncorrected"].append(
                    abs(pe_group_difference['uncorrected'] - pe_group_difference['unbiased']))
                oracle_measures["pe"]["corrected"].append(
                    abs(pe_group_difference['corrected'] - pe_group_difference['unbiased']))
                oracle_measures["pe"]["corrected_l"].append(
                    abs((pe_bias_withLabelled[0] - pe_bias_withLabelled[1]) - pe_group_difference['unbiased']))
                oracle_measures["pe"]["corrected_u_l"].append(
                    abs((pe_bias_withUL[0] - pe_bias_withUL[1]) - pe_group_difference['unbiased']))

                # predictive positive value = TPR*alpha / (TPR*alpha + FPR*(1-alpha))
                ppv_group_difference = correct_metrics.correct_ppv(eo, pe, alpha, alpha_)
                ppv_bias_withLabelled = {g: eo_bias_withLabelled[g] * alpha[g] / (
                            eo_bias_withLabelled[g] * alpha[g] + pe_bias_withLabelled[g] * (1 - alpha[g]))
                                         for g in range(num_groups)}
                ppv_bias_withUL = {g: eo_bias_withUL[g] * alpha[g] / (
                            eo_bias_withUL[g] * alpha[g] + pe_bias_withUL[g] * (1 - alpha[g]))
                                   for g in range(num_groups)}

                oracle_measures["ppv"]["uncorrected"].append(
                    abs(ppv_group_difference['uncorrected'] - ppv_group_difference['unbiased']))
                oracle_measures["ppv"]["corrected"].append(
                    abs(ppv_group_difference['corrected'] - ppv_group_difference['unbiased']))
                oracle_measures["ppv"]["corrected_l"].append(
                    abs((ppv_bias_withLabelled[0] - ppv_bias_withLabelled[1]) - ppv_group_difference['unbiased']))
                oracle_measures["ppv"]["corrected_u_l"].append(
                    abs((ppv_bias_withUL[0] - ppv_bias_withUL[1]) - ppv_group_difference['unbiased']))

                if args.estimate:

                    Kfit = [comp, comp]
                    nested_group_EM = NestedGroupDist(x_unlabeled=s_ub,
                                                      x_labeled=[s_pos_bias, s_neg_bias],
                                                      unlabeled_groups=np.hstack([np.zeros(s1_ub.shape[0]),
                                                                                  np.ones(s2_ub.shape[0])]),
                                                      labeled_groups=[np.hstack([np.zeros(s1_pos_bias.shape[0]),
                                                                                 np.ones(s2_pos_bias.shape[0])]),
                                                                      np.hstack([np.zeros(s1_neg_bias.shape[0]),
                                                                                 np.ones(s2_neg_bias.shape[0])])],
                                                      components=Kfit, num_classes=2, num_groups=2)
                    nested_group_EM.estimate_params(max_steps=5000)

                    alphas_1_est, alphas_2_est = nested_group_EM.alphas
                    w_1_est = [nested_group_EM.w[c][0] for c in [0, 1]]  # group==0 for each class
                    wl_1_est = [nested_group_EM.w_labeled[c][0] for c in [0, 1]]
                    w_2_est = [nested_group_EM.w[c][1] for c in [0, 1]]  # group==1 for each class
                    wl_2_est = [nested_group_EM.w_labeled[c][1] for c in [0, 1]]
                    lls_1 = lls_2 = nested_group_EM.lls

                    # Redefining mixtures with estimated parameters.
                    # Distributions with estimated parameters are used for all subsequent operations
                    estimated_d_unbiased = {
                        0: {"pos": NMixture(nested_group_EM.mu[0], nested_group_EM.sg[0], np.array(w_1_est[0])),
                            "neg": NMixture(nested_group_EM.mu[1], nested_group_EM.sg[1], np.array(w_1_est[1]))},
                        1: {"pos": NMixture(nested_group_EM.mu[0], nested_group_EM.sg[0], np.array(w_2_est[0])),
                            "neg": NMixture(nested_group_EM.mu[1], nested_group_EM.sg[1], np.array(w_2_est[1]))}}

                    estimated_d_biased = {
                        0: {"pos": NMixture(nested_group_EM.mu[0], nested_group_EM.sg[0], np.array(wl_1_est[0])),
                            "neg": NMixture(nested_group_EM.mu[1], nested_group_EM.sg[1], np.array(wl_1_est[1]))},
                        1: {"pos": NMixture(nested_group_EM.mu[0], nested_group_EM.sg[0], np.array(wl_2_est[0])),
                            "neg": NMixture(nested_group_EM.mu[1], nested_group_EM.sg[1], np.array(wl_2_est[1]))}}

                    # Redefining alpha with the estimated parameter
                    alpha_est = [alphas_1_est[0], alphas_2_est[0]]
                    # observed alpha
                    alpha_l_est = [s1_pos_bias.shape[0] / sum([s1_pos_bias.shape[0], s1_neg_bias.shape[0]]),
                                   s2_pos_bias.shape[0] / sum([s2_pos_bias.shape[0], s2_neg_bias.shape[0]])]

                    # estimated biases
                    # for a more stable AUC computation, we average 5 estimations of AUC
                    num_auc_runs = 5
                    for g in range(num_groups):
                        estimated_measures["bias"][f"aucpp_g{g + 1}"].append(
                            np.mean([mixtureUtils.auc(estimated_d_unbiased[g]["pos"], estimated_d_biased[g]["pos"])
                                     for _ in range(num_auc_runs)]))
                        estimated_measures["bias"][f"aucnn_g{g + 1}"].append(
                            np.mean([mixtureUtils.auc(estimated_d_unbiased[g]["neg"], estimated_d_biased[g]["neg"])
                                     for _ in range(num_auc_runs)]))
                        estimated_measures["bias"][f"aucpn_g{g + 1}"].append(
                            np.mean([mixtureUtils.auc(estimated_d_unbiased[g]["pos"], estimated_d_unbiased[g]["neg"])
                                     for _ in range(num_auc_runs)]))

                    # Unbiased group distribution
                    estimated_p = {
                        0: PUMixture(estimated_d_unbiased[0]['pos'], estimated_d_unbiased[0]['neg'], alpha_est[0]),
                        1: PUMixture(estimated_d_unbiased[1]['pos'], estimated_d_unbiased[1]['neg'], alpha_est[1])}

                    # Biased group distribution
                    estimated_q = {
                        0: PUMixture(estimated_d_biased[0]['pos'], estimated_d_biased[0]['neg'], alpha_l_est[0]),
                        1: PUMixture(estimated_d_biased[1]['pos'], estimated_d_biased[1]['neg'], alpha_l_est[1])}

                    # Create a distribution object for the entire unblabeled set (not conditioned on a group)
                    est_alpha_u = {1: sum(eta[g] * alpha_est[g] for g in range(num_groups)),
                                   0: sum(eta[g] * (1 - alpha_est[g]) for g in range(num_groups))}
                    est_wPos_u = sum(eta[g] * alpha_est[g] * estimated_d_unbiased[g]['pos'].ps for g in
                                     range(num_groups)) / est_alpha_u[1]
                    est_wNeg_u = sum(eta[g] * (1 - alpha_est[g]) * estimated_d_unbiased[g]['neg'].ps for g in
                                     range(num_groups)) / est_alpha_u[0]
                    est_d_pos_ub = NMixture(nested_group_EM.mu[0], nested_group_EM.sg[0], est_wPos_u)
                    est_d_neg_ub = NMixture(nested_group_EM.mu[1], nested_group_EM.sg[1], est_wNeg_u)
                    est_d_ub = PUMixture(est_d_pos_ub, est_d_neg_ub, est_alpha_u[1])

                    # Create a distribution object for the entire labeled set (not conditioned on a group)
                    # wPos_l and wNeg_l can be estimated directly from a different fromulation of the GMM estimation,
                    # where the labeled data is treated as one sample
                    # instead of two samples (one for each group).
                    est_alpha_l = {1: sum(eta_[g] * alpha_l_est[g] for g in range(num_groups)),
                                   0: sum(eta_[g] * (1 - alpha_l_est[g]) for g in range(num_groups))}
                    est_wPos_l = sum(
                        eta_[g] * alpha_l_est[g] * estimated_d_biased[g]['pos'].ps for g in range(num_groups)) / \
                                 est_alpha_l[1]
                    est_wNeg_l = sum(eta_[g] * (1 - alpha_l_est[g]) * estimated_d_biased[g]['neg'].ps for g in
                                     range(num_groups)) / est_alpha_l[0]

                    est_d_pos_bias = NMixture(nested_group_EM.mu[0], nested_group_EM.sg[0], est_wPos_l)
                    est_d_neg_bias = NMixture(nested_group_EM.mu[1], nested_group_EM.sg[1], est_wNeg_l)
                    est_d_bias = PUMixture(est_d_pos_bias, est_d_neg_bias, est_alpha_l[1])

                    if classifier_threshold == 'fixed':
                        class_threshold = 0.5
                    else:
                        class_threshold = est_alpha_l[1]


                    YSoft_u = est_d_ub.pn_posterior(s_ub).reshape(-1, 1)
                    # Since we do not have labels from the unlabeled data,
                    # we use the posterior probability from the GMM as soft labels
                    Y = np.concatenate([label_all_bias, YSoft_u])
                    YPred = np.concatenate([y_all_bias, y_all_ub])
                    WPos_l = {g: estimated_d_unbiased[g]["pos"].points_pdf(s_all_bias) / est_d_pos_bias.points_pdf(s_all_bias)
                              for g in range(num_groups)}
                    WPos_u = {g: estimated_d_unbiased[g]["pos"].points_pdf(s_ub) / est_d_pos_ub.points_pdf(s_ub)
                              for g in range(num_groups)}
                    WNeg_l = {g: estimated_d_unbiased[g]["neg"].points_pdf(s_all_bias) / est_d_neg_bias.points_pdf(
                        s_all_bias) for g in range(num_groups)}
                    WNeg_u = {g: estimated_d_unbiased[g]["neg"].points_pdf(s_ub) / est_d_neg_ub.points_pdf(s_ub)
                              for g in range(num_groups)}

                    TPR = {g: correct_metrics.TPR(Y, YPred, class_threshold, np.concatenate((WPos_l[g], WPos_u[g])))
                           for g in range(num_groups)}
                    FPR = {g: correct_metrics.FPR(Y, YPred, class_threshold, np.concatenate((WNeg_l[g], WNeg_u[g])))
                           for g in range(num_groups)}

                    dp_group_difference, _ = correct_metrics.correct_dp(points_bias={0: s1_bias, 1: s2_bias},
                                                                        y_all_bias={0: y1_all_bias, 1: y2_all_bias},
                                                                        y_all_ub={0: y1_all_ub, 1: y2_all_ub},
                                                                        p=estimated_p,
                                                                        q=estimated_q)
                    dp_bias_withLabelled = {g: sum(
                        y_all_bias * estimated_p[g].points_pdf(s_all_bias) / est_d_bias.points_pdf(s_all_bias)) / \
                                             y_all_bias.shape[0]
                                              for g in range(num_groups)}
                    dp_bias_withUL = {g: TPR[g] + FPR[g] for g in range(num_groups)}

                    estimated_measures["dp"]["uncorrected"].append(
                        abs(dp_group_difference['uncorrected'] - dp_group_difference['unbiased']))
                    estimated_measures["dp"]["corrected"].append(
                        abs(dp_group_difference['corrected'] - dp_group_difference['unbiased']))
                    estimated_measures["dp"]["corrected_l"].append(
                        abs((dp_bias_withLabelled[0] - dp_bias_withLabelled[1]) - dp_group_difference['unbiased']))
                    estimated_measures["dp"]["corrected_u_l"].append(
                        abs((dp_bias_withUL[0] - dp_bias_withUL[1]) - dp_group_difference['unbiased']))

                    ## Equal opportunity
                    eo_group_difference, eo = correct_metrics.correct_eo(
                        points_pos_bias={0: s1_pos_bias, 1: s2_pos_bias},
                        y_pos_bias={0: y1_pos_bias, 1: y2_pos_bias},
                        y_pos_ub={0: y1_pos_ub, 1: y2_pos_ub},
                        d_pos_bias={0: estimated_d_biased[0]["pos"], 1: estimated_d_biased[1]["pos"]},
                        d_pos_ub={0: estimated_d_unbiased[0]["pos"], 1: estimated_d_unbiased[1]["pos"]})

                    eo_bias_withLabelled = {
                        g: sum(
                            y_pos_bias * estimated_d_unbiased[g]["pos"].points_pdf(s_pos_bias) / est_d_pos_bias.points_pdf(
                            s_pos_bias)) / y_pos_bias.shape[0] for g in range(num_groups)}
                    eo_bias_withUL = {g: TPR[g] for g in range(num_groups)}

                    estimated_measures["eo"]["uncorrected"].append(
                        abs(eo_group_difference['uncorrected'] - eo_group_difference['unbiased']))
                    estimated_measures["eo"]["corrected"].append(
                        abs(eo_group_difference['corrected'] - eo_group_difference['unbiased']))
                    estimated_measures["eo"]["corrected_l"].append(
                        abs((eo_bias_withLabelled[0] - eo_bias_withLabelled[1]) - eo_group_difference['unbiased']))
                    estimated_measures["eo"]["corrected_u_l"].append(
                        abs((eo_bias_withUL[0] - eo_bias_withUL[1]) - eo_group_difference['unbiased']))

                    ## predictive equality
                    pe_group_difference, pe = correct_metrics.correct_pe(
                        points_neg_bias={0: s1_neg_bias, 1: s2_neg_bias},
                        y_neg_bias={0: y1_neg_bias, 1: y2_neg_bias},
                        y_neg_ub={0: y1_neg_ub, 1: y2_neg_ub},
                        d_neg_bias={0: estimated_d_biased[0]["neg"], 1: estimated_d_biased[1]["neg"]},
                        d_neg_ub={0: estimated_d_unbiased[0]["neg"], 1: estimated_d_unbiased[1]["neg"]})
                    pe_bias_withLabelled = {
                        g: sum(
                            y_neg_bias * estimated_d_unbiased[g]["neg"].points_pdf(s_neg_bias) / est_d_neg_bias.points_pdf(
                            s_neg_bias)) / y_neg_bias.shape[0] for g in range(num_groups)}
                    pe_bias_withUL = {g: FPR[g] for g in range(num_groups)}

                    estimated_measures["pe"]["uncorrected"].append(
                        abs(pe_group_difference['uncorrected'] - pe_group_difference['unbiased']))
                    estimated_measures["pe"]["corrected"].append(
                        abs(pe_group_difference['corrected'] - pe_group_difference['unbiased']))
                    estimated_measures["pe"]["corrected_l"].append(
                        abs((pe_bias_withLabelled[0] - pe_bias_withLabelled[1]) - pe_group_difference['unbiased']))
                    estimated_measures["pe"]["corrected_u_l"].append(
                        abs((pe_bias_withUL[0] - pe_bias_withUL[1]) - pe_group_difference['unbiased']))

                    # predictive positive value = TPR*alpha / (TPR*alpha + FPR*(1-alpha))
                    ppv_group_difference = correct_metrics.correct_ppv(eo, pe, alpha_est, alpha_l_est)
                    ppv_bias_withLabelled = {g:  eo_bias_withLabelled[g] * alpha_est[g] / (
                                eo_bias_withLabelled[g] * alpha_est[g] + pe_bias_withLabelled[g] * (1 - alpha_est[g]))
                                             for g in range(num_groups)}
                    ppv_bias_withUL = {g: eo_bias_withUL[g] * alpha_est[g] / (
                                eo_bias_withUL[g] * alpha_est[g] + pe_bias_withUL[g] * (1 - alpha_est[g]))
                                       for g in range(num_groups)}

                    estimated_measures["ppv"]["uncorrected"].append(
                        abs(ppv_group_difference['uncorrected'] - ppv_group_difference['unbiased']))
                    estimated_measures["ppv"]["corrected"].append(
                        abs(ppv_group_difference['corrected'] - ppv_group_difference['unbiased']))
                    estimated_measures["ppv"]["corrected_l"].append(
                        abs((ppv_bias_withLabelled[0] - ppv_bias_withLabelled[1]) - ppv_group_difference['unbiased']))
                    estimated_measures["ppv"]["corrected_u_l"].append(
                        abs((ppv_bias_withUL[0] - ppv_bias_withUL[1]) - ppv_group_difference['unbiased']))

                if args.estimateNG:

                    Kfit = [comp, comp]
                    nested_group_EM = NestedGroupDistUnknownGroup(
                        x_unlabeled=s_ub, x_labeled=[s_pos_bias, s_neg_bias],
                        unlabeled_groups=np.hstack([np.zeros(s1_ub.shape[0]), np.ones(s2_ub.shape[0])]),
                        components=Kfit, num_classes=2, num_groups=2)
                    nested_group_EM.estimate_params(max_steps=5000)

                    alphas_1_est, alphas_2_est = nested_group_EM.alphas
                    w_1_est = [nested_group_EM.w[c][0] for c in [0, 1]]  # group==0 for each class
                    wl_est = [nested_group_EM.w_labeled[c] for c in [0, 1]]
                    w_2_est = [nested_group_EM.w[c][1] for c in [0, 1]]  # group==1 for each class
                    lls_1 = lls_2 = nested_group_EM.lls

                    # Redefining mixtures with estimated parameters.
                    # Distributions with estimated parameters are used for all subsequent operations
                    estimated_d_unbiased = {
                        0: {"pos": NMixture(nested_group_EM.mu[0], nested_group_EM.sg[0], np.array(w_1_est[0])),
                            "neg": NMixture(nested_group_EM.mu[1], nested_group_EM.sg[1], np.array(w_1_est[1]))},
                        1: {"pos": NMixture(nested_group_EM.mu[0], nested_group_EM.sg[0], np.array(w_2_est[0])),
                            "neg": NMixture(nested_group_EM.mu[1], nested_group_EM.sg[1], np.array(w_2_est[1]))}}

                    estimated_d_biased = {
                        "pos": NMixture(nested_group_EM.mu[0], nested_group_EM.sg[0], np.array(wl_est[0])),
                        "neg": NMixture(nested_group_EM.mu[1], nested_group_EM.sg[1], np.array(wl_est[1]))}

                    # Redefining alpha with the estimated parameter
                    alpha_est = [alphas_1_est[0], alphas_2_est[0]]
                    # observed alpha
                    alpha_l_est = s_pos_bias.shape[0] / sum([s_pos_bias.shape[0], s_neg_bias.shape[0]])

                    # estimated biases
                    # for a more stable AUC computation, we average 5 estimations of AUC
                    num_auc_runs = 5
                    for g in range(num_groups):
                        estimated_measures_ng["bias"][f"aucpp_g{g + 1}"].append(
                            np.mean([mixtureUtils.auc(estimated_d_unbiased[g]["pos"], estimated_d_biased["pos"])
                                     for _ in range(num_auc_runs)]))
                        estimated_measures_ng["bias"][f"aucnn_g{g + 1}"].append(
                            np.mean([mixtureUtils.auc(estimated_d_unbiased[g]["neg"], estimated_d_biased["neg"])
                                     for _ in range(num_auc_runs)]))
                        estimated_measures_ng["bias"][f"aucpn_g{g + 1}"].append(
                            np.mean([mixtureUtils.auc(estimated_d_unbiased[g]["pos"], estimated_d_unbiased[g]["neg"])
                                     for _ in range(num_auc_runs)]))

                    # Unbiased group distribution
                    estimated_p = {
                        0: PUMixture(estimated_d_unbiased[0]['pos'], estimated_d_unbiased[0]['neg'], alpha_est[0]),
                        1: PUMixture(estimated_d_unbiased[1]['pos'], estimated_d_unbiased[1]['neg'], alpha_est[1])}

                    # Biased group distribution
                    estimated_q = PUMixture(estimated_d_biased['pos'], estimated_d_biased['neg'], alpha_l_est)

                    # Create a distribution object for the entire unlabeled set (not conditioned on a group)
                    est_alpha_u = {1: sum(eta[g] * alpha_est[g] for g in range(num_groups)),
                                   0: sum(eta[g] * (1 - alpha_est[g]) for g in range(num_groups))}
                    est_wPos_u = sum(eta[g] * alpha_est[g] * estimated_d_unbiased[g]['pos'].ps for g in
                                     range(num_groups)) / est_alpha_u[1]
                    est_wNeg_u = sum(eta[g] * (1 - alpha_est[g]) * estimated_d_unbiased[g]['neg'].ps for g in
                                     range(num_groups)) / est_alpha_u[0]
                    est_d_pos_ub = NMixture(nested_group_EM.mu[0], nested_group_EM.sg[0], est_wPos_u)
                    est_d_neg_ub = NMixture(nested_group_EM.mu[1], nested_group_EM.sg[1], est_wNeg_u)
                    est_d_ub = PUMixture(est_d_pos_ub, est_d_neg_ub, est_alpha_u[1])

                    # Create a distribution object for the entire labeled set (not conditioned on a group)
                    # wPos_l and wNeg_l can be estimated directly from a different fromulation of the GMM estimation,
                    # where the labeled data is treated as one sample
                    # instead of two samples (one for each group).
                    est_wPos_l = estimated_d_biased['pos'].ps
                    est_wNeg_l = estimated_d_biased['neg'].ps

                    est_d_pos_bias = estimated_d_biased['pos']
                    est_d_neg_bias = estimated_d_biased['neg']

                    if classifier_threshold == 'fixed':
                        class_threshold = 0.5
                    else:
                        class_threshold = alpha_l_est

                    
                    YSoft_u = est_d_ub.pn_posterior(s_ub).reshape(-1, 1)
                    # Since we do not have labels from the unlabeled data,
                    # we use the posterior probability from the GMM as soft labels
                    Y = np.concatenate([label_all_bias, YSoft_u])
                    YPred = np.concatenate([y_all_bias, y_all_ub])
                    WPos_l = {
                        g: estimated_d_unbiased[g]["pos"].points_pdf(s_all_bias) / est_d_pos_bias.points_pdf(s_all_bias)
                        for g in range(num_groups)}
                    WPos_u = {g: estimated_d_unbiased[g]["pos"].points_pdf(s_ub) / est_d_pos_ub.points_pdf(s_ub)
                              for g in range(num_groups)}
                    WNeg_l = {g: estimated_d_unbiased[g]["neg"].points_pdf(s_all_bias) / est_d_neg_bias.points_pdf(
                        s_all_bias) for g in range(num_groups)}
                    WNeg_u = {g: estimated_d_unbiased[g]["neg"].points_pdf(s_ub) / est_d_neg_ub.points_pdf(s_ub)
                              for g in range(num_groups)}

                    TPR = {g: correct_metrics.TPR(Y, YPred, class_threshold, np.concatenate((WPos_l[g], WPos_u[g])))
                           for g in range(num_groups)}
                    FPR = {g: correct_metrics.FPR(Y, YPred, class_threshold, np.concatenate((WNeg_l[g], WNeg_u[g])))
                           for g in range(num_groups)}

                    dp_group_difference, _ = correct_metrics_unknowngroups.correct_dp(y_all_bias=y_all_bias,
                                                                                      y_all_ub={0: y1_all_ub, 1: y2_all_ub},
                                                                                      s_all_bias=s_all_bias,
                                                                                      p=estimated_p, q=estimated_q
                                                                                      )
                    dp_bias_withUL = {g: TPR[g] + FPR[g] for g in range(num_groups)}

                    estimated_measures_ng["dp"]["corrected_l"].append(
                        abs(dp_group_difference['corrected_l'] - dp_group_difference['unbiased']))
                    estimated_measures_ng["dp"]["corrected_u_l"].append(
                        abs((dp_bias_withUL[0] - dp_bias_withUL[1]) - dp_group_difference['unbiased']))

                    ## Equal opportunity
                    eo_group_difference, eo = correct_metrics_unknowngroups.correct_eo(
                        y_pos_bias=y_pos_bias,
                        y_pos_ub={0: y1_pos_ub, 1: y2_pos_ub},
                        s_pos_bias=s_pos_bias,
                        q_pos=estimated_d_biased["pos"],
                        p_pos={0: estimated_d_unbiased[0]["pos"], 1: estimated_d_unbiased[1]["pos"]})

                    eo_bias_withUL = {g: TPR[g] for g in range(num_groups)}

                    estimated_measures_ng["eo"]["corrected_l"].append(
                        abs(eo_group_difference['corrected_l'] - eo_group_difference['unbiased']))
                    estimated_measures_ng["eo"]["corrected_u_l"].append(
                        abs((eo_bias_withUL[0] - eo_bias_withUL[1]) - eo_group_difference['unbiased']))

                    ## predictive equality
                    pe_group_difference, pe = correct_metrics_unknowngroups.correct_pe(
                        y_neg_bias=y_neg_bias,
                        y_neg_ub={0: y1_neg_ub, 1: y2_neg_ub},
                        s_neg_bias=s_neg_bias,
                        q_neg=estimated_d_biased["neg"],
                        p_neg={0: estimated_d_unbiased[0]["neg"], 1: estimated_d_unbiased[1]["neg"]})

                    pe_bias_withUL = {g: FPR[g] for g in range(num_groups)}
                    estimated_measures_ng["pe"]["corrected_l"].append(
                        abs(pe_group_difference['corrected_l'] - pe_group_difference['unbiased']))
                    estimated_measures_ng["pe"]["corrected_u_l"].append(
                        abs((pe_bias_withUL[0] - pe_bias_withUL[1]) - pe_group_difference['unbiased']))

                    # predictive positive value = TPR*alpha / (TPR*alpha + FPR*(1-alpha))
                    ppv_group_difference = correct_metrics_unknowngroups.correct_ppv(
                        eo, pe, alpha_est,  eo_bias_withUL, pe_bias_withUL)

                    estimated_measures_ng["ppv"]["corrected_l"].append(
                        abs(ppv_group_difference['corrected_l'] - ppv_group_difference['unbiased']))
                    estimated_measures_ng["ppv"]["corrected_u_l"].append(
                        abs(ppv_group_difference['corrected_u_l'] - ppv_group_difference['unbiased']))



                # Save fairness metrics data as npz files for each dimension
                np.savez(file_path, **{"oracle": oracle_measures, "estimated": estimated_measures, 'estimated_ng': estimated_measures_ng})
            
            if save_model_params and nested_group_EM is not None:
                with open(file_path.split('.npz')[0] + '_gmm_model.pkl', 'wb') as f:
                    run_intermediate = {'index': i, 'EMmodel': nested_group_EM}

                    pickle.dump(run_intermediate, f)
