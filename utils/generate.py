import random
from utils.Mixture import NMixture, PUMixture
import numpy as np

def generateSamples(d1_pos_ub, d1_neg_ub, d2_pos_ub, d2_neg_ub, d1_pos_bias, d1_neg_bias, \
                    d2_pos_bias, d2_neg_bias, total_samples, lam, alpha, alpha_,eta,eta_):
    
    def round_to_int(x):
        return np.round(x, 0).astype(int).item()

    # Unbiased samples
    N_ub = round_to_int(total_samples * (1-lam))

    s1_pos_ub = d1_pos_ub.sample_points( round_to_int(N_ub * eta[0] * alpha[0]) )
    s1_neg_ub = d1_neg_ub.sample_points( round_to_int(N_ub * eta[0] * (1-alpha[0])) )
    s2_pos_ub = d2_pos_ub.sample_points( round_to_int(N_ub * eta[1] * alpha[1]) )
    s2_neg_ub = d2_neg_ub.sample_points( round_to_int(N_ub * eta[1] * (1-alpha[1])) )

    # Biased samples
    N_bias =  total_samples - N_ub

    s1_pos_bias = d1_pos_bias.sample_points( round_to_int(N_bias * eta_[0] * alpha_[0]) )
    s1_neg_bias = d1_neg_bias.sample_points( round_to_int(N_bias * eta_[0] * (1-alpha_[0])) )
    s2_pos_bias = d2_pos_bias.sample_points( round_to_int(N_bias * eta_[1] * alpha_[1]) )
    s2_neg_bias = d2_neg_bias.sample_points( round_to_int(N_bias * eta_[1] * (1-alpha_[1])) )

    return s1_pos_ub, s1_neg_ub, s2_pos_ub, s2_neg_ub, s1_pos_bias, \
        s1_neg_bias, s2_pos_bias, s2_neg_bias
                
