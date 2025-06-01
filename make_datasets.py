import os
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse

from pnuem.Mixture import NMixture, PUMixture
from pnuem.generate import generateSamples
import pnuem.mixtureUtils as mixtureUtils
import pnuem.correct_metrics as correct_metrics

from pnuem.NNclassifier import getModel, getModelScores

from pnuem.NestedGroupDist import NestedGroupDist



total_samples = 25000  # Total number of data samples
THREADS=8
classifier_threshold = "fixed"
num_groups=2


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser('get_fairness_metrics')
    parser.add_argument("--dim", type=int, help='dimension')
    parser.add_argument("--k", type=int, help='number of components')
    parser.add_argument("--lam", type=float, help='lambda value, usually 0.5 or 0.1')
    parser.add_argument("--eta", type=float, help='eta value, usually 0.5 or 0.1')
    #parser.add_argument("--output", type=str, default=".",
    #                    help="Path to file where the metrics are to be stored")

    args = parser.parse_args()
    return args 

def generate_dataset_file(datas, dim, K, idx, model):
    synth = []

    for name, data  in datas.items():
        points = pd.DataFrame(data)
        points['group']=int(name.split('_')[0][-1])-1
        points['label']=int('pos' in name)
        points['bias'] ='bias' in name
        points['scores'] = getModelScores(model, data)
        synth.append(points)
    ds = pd.concat(synth, ignore_index=True)

    return ds

def save_dataset_file(df, dim, K, idx, aucbucket, lam, eta):
    df.drop(columns='scores').to_csv(f'resultsSynth/synthd{dim}K{K}A{aucbucket}_{lam}_{eta}_{idx}_numerical.csv')
    filename = f'resultsSynth/synthd{dim}K{K}A{aucbucket}_{lam}_{eta}_{idx}_group_scores_remapped.csv'
    pd.DataFrame({'group': df['group'].astype(int),
                  'label': df['label'].astype(int),
                  'score_MultiLayerPerceptron':df['scores']}).to_csv(filename)




def make_datasets(df, aucbucket, 
                  lam=0.5, alpha=[0.5, 0.5], alpha_=[0.5, 0.5], eta=[0.5, 0.5], eta_=[0.5, 0.5]):

    oracle_measures = {measure: {"uncorrected": list(),
                             "corrected": list(),
                             "corrected_l": list(),
                             "corrected_u_l": list()} for measure in ['dp', 'eo', 'pe', 'ppv']}
    oracle_measures["eo_group_difference"] =  list()
    oracle_measures["pe_group_difference"] =  list()
    oracle_measures["ppv_group_difference"] =  list()

    oracle_measures["bias"] = {"aucpn_g1": list(), "aucpn_g2": list(),
                               "aucpp_g1": list(), "aucpp_g2": list(),
                               "aucnn_g1": list(), "aucnn_g2": list(),
                               "avg_sb": list(),
                               "alpha": alpha, 'alpha_l': alpha_, 'lambda': lam}  # save setting
    
    dim = df['dim'].unique()[0].item()
    comp = df['K'].unique()[0].item()
    
    for i in tqdm(range(df.shape[0])):
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
    
        # Generate samples according to the distributions
        s1_pos_ub, s1_neg_ub, s2_pos_ub, s2_neg_ub, s1_pos_bias, \
            s1_neg_bias, s2_pos_bias, s2_neg_bias \
            = generateSamples(distrib_ub[0][1], distrib_ub[0][0],
                              distrib_ub[1][1], distrib_ub[1][0],
                              distrib_bias[0][1], distrib_bias[0][0],
                              distrib_bias[1][1], distrib_bias[1][0],
                              total_samples, lam, alpha, alpha_, eta, eta_)
    
        
        
        # Positive and negative labeled samples
        s_pos_bias = np.concatenate((s1_pos_bias, s2_pos_bias), axis=0)
        s_neg_bias = np.concatenate((s1_neg_bias, s2_neg_bias), axis=0)
    
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
    
        # print(f'{i} avg bias:{oracle_measures["bias"]["avg_sb"][0]:.4f}')
    
        ################### Train Classifier ################
        # Using the biased samples to train the model
        X = np.vstack((s_pos_bias, s_neg_bias))
        y = np.hstack((np.ones(len(s_pos_bias)), np.zeros(len(s_neg_bias))))
    
        # Get the trained model on this data
        model = getModel(X, y, threads=THREADS)    
    
        # save to file 
        datas= {"s1_pos_bias": s1_pos_bias, 
            "s1_neg_bias": s1_neg_bias,
            "s2_pos_bias": s2_pos_bias,
            "s2_neg_bias": s2_neg_bias,
            "s1_pos_ub": s1_pos_ub, 
            "s1_neg_ub": s1_neg_ub, 
            "s2_pos_ub": s2_pos_ub, 
            "s2_neg_ub": s2_neg_ub}
        ds = generate_dataset_file(datas, dim, comp, i, model)
        save_dataset_file(ds, dim, comp, i, aucbucket, lam, eta[0])
  
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
      
        if classifier_threshold == 'fixed':
            class_threshold = 0.5
        else:
             class_threshold = alpha_l[1]
  
        y_bias = {'pos': {0: getModelScores(model, s1_pos_bias),
                      1: getModelScores(model, s2_pos_bias)},
              'neg': {0: getModelScores(model, s1_neg_bias),
                      1: getModelScores(model, s2_neg_bias)}}
        y_ub = {'pos': {0: getModelScores(model, s1_pos_ub),
                        1: getModelScores(model, s2_pos_ub)},
                'neg': {0: getModelScores(model, s1_neg_ub),
                        1: getModelScores(model, s2_neg_ub)}}
        y1_all_bias = getModelScores(model, s1_bias) >= class_threshold
        y1_pos_bias = y_bias['pos'][0] >= class_threshold
        y1_neg_bias = y_bias['neg'][0] >= class_threshold
        y2_all_bias = getModelScores(model, s2_bias) >= class_threshold
        y2_pos_bias = y_bias['pos'][1] >= class_threshold
        y2_neg_bias = y_bias['neg'][1] >= class_threshold
        y_all_bias = np.concatenate([y1_all_bias, y2_all_bias])
    
        y1_all_ub = getModelScores(model, s1_ub) >= class_threshold
        y1_pos_ub = y_ub['pos'][0] >= class_threshold
        y1_neg_ub = y_ub['neg'][0] >= class_threshold
        y2_all_ub = getModelScores(model, s2_ub) >= class_threshold
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



        # Compute TPR and FPR for groups for Correction with labeled and unlabeled points

        # Since we do not have labels from the unlabeled data,
        # we use the posterior probability from the GMM as soft labels
        YSoft_u = d_ub.pn_posterior(s_ub).reshape(-1, 1)
        # print(YSoft_u.shape)
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
    
        # group_oracle_measures = {'eo':eo, 'pe':pe}
        oracle_measures['eo_group_difference'].append(eo_group_difference)
        oracle_measures['pe_group_difference'].append(pe_group_difference)
        oracle_measures['ppv_group_difference'].append(ppv_group_difference)
        #oracle_differences = {'eo_group_difference':eo_group_difference,
        #                      'pe_group_difference': pe_group_difference,
        #                      'ppv_group_difference': ppv_group_difference}
     
    return oracle_measures

def main(): 
    args = parse_args()
    dim = args.dim
    comp = args.k
    lam = args.lam
    
    if args.eta==0.5:
        etas = [0.5, 0.5]
        etas_ = [0.5, 0.5]
    else:
        etas = [0.1, 0.9]
        etas_ = [0.05, 0.95]

    aucpn_ranges = [[0.65, 0.75], [0.75, 0.85], [0.85, 0.95]]

    print(f'd{dim}K{comp} lambda={lam} eta={etas}')
    pkl_file_name = 'synthetic_param/synthetic_datasets_params_d{}K{}.pkl'.format(dim, comp)
    params = pd.read_pickle(pkl_file_name)
    
    orc_measures = []

    for au, aucpn_range in enumerate(aucpn_ranges):
        df = params[(params['aucpn'] > aucpn_range[0]) & (params['aucpn'] < aucpn_range[1])]
        orc=make_datasets(df, au, lam, alpha=[0.5, 0.5], alpha_=[0.5,0.5],
                          eta=etas, eta_=etas_)

        orc_measures.append([{'dim':dim, 'K':comp, 'auc':au, 'measures':orc}])

    oracle_measures = pd.DataFrame(orc_measures)
    oracle_measures.to_pickle(f'synthetic_param/oracle_d{dim}K{comp}_lam{lam}_eta{args.eta}.pkl')

   


if __name__ == '__main__':
    main()


