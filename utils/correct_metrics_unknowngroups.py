import numpy as np


def correct_dp(y_all_bias, y_all_ub, s_all_bias, p, q):
    # Demographic parity
    groups = y_all_ub.keys()

    dp_ub = {g : sum(y_all_ub[g]) / y_all_ub[g].shape[0] for g in groups}
    dp_bias_withLabelled = {
        g: sum(y_all_bias * p[g].points_pdf(s_all_bias) / q.points_pdf(s_all_bias)) / y_all_bias.shape[0] for g in groups}

    dp_group_difference = {'unbiased': dp_ub[0] - dp_ub[1],
                           'corrected_l': dp_bias_withLabelled[0] - dp_bias_withLabelled[1]}

    return dp_group_difference, {'dp_ub': dp_ub,
                                 'dp_bias_withLabelled': dp_bias_withLabelled, }


def correct_eo(y_pos_bias, y_pos_ub, s_pos_bias, p_pos, q_pos):
    # Equal opportunity
    groups = list(y_pos_ub.keys())

    if y_pos_ub[groups[0]] is not None:
        eo_ub = {g: sum(y_pos_ub[g]) / y_pos_ub[g].shape[0] for g in groups}
    else:
        eo_ub = {g: 0 for g in groups}

    eo_bias_withLabelled = {
        g: sum(y_pos_bias * p_pos[g].points_pdf(
                s_pos_bias) / q_pos.points_pdf(
                s_pos_bias)) / y_pos_bias.shape[0] for g in groups}

    eo_group_difference = {'unbiased': eo_ub[0] - eo_ub[1],
                           'corrected_l': eo_bias_withLabelled[0] - eo_bias_withLabelled[1]}

    return eo_group_difference, {'eo_ub': eo_ub, 'eo_bias_withLabelled': eo_bias_withLabelled}


def correct_pe(y_neg_bias, y_neg_ub, s_neg_bias, p_neg, q_neg):
    # predictive equality
    groups = list(y_neg_ub.keys())
    if y_neg_ub[groups[0]] is not None:
        pe_ub = {g: sum(y_neg_ub[g]) / y_neg_ub[g].shape[0] for g in groups}
    else:
        pe_ub = {g: 0 for g in groups}

    pe_bias_withLabelled = {
        g: sum(y_neg_bias * p_neg[g].points_pdf(s_neg_bias) / q_neg.points_pdf(s_neg_bias)) / y_neg_bias.shape[0] for g in groups}

    pe_group_difference = {'unbiased': pe_ub[0] - pe_ub[1],
                           'corrected_l': pe_bias_withLabelled[0] - pe_bias_withLabelled[1]}
    return pe_group_difference, {'pe_ub': pe_ub, 'pe_bias_withLabelled': pe_bias_withLabelled}


def correct_ppv(eo, pe, alpha):
    # predictive positive value = TPR*alpha / (TPR*alpha + FPR*(1-alpha))

    groups= eo['eo_ub'].keys()
    ppv_ub = {g: (eo['eo_ub'][g] * alpha[g]) / (eo['eo_ub'][g] * alpha[g] + pe['pe_ub'][g] * (1 - alpha[g]))
              for g in groups}
    ppv_bias_withLabelled = {g: eo['eo_bias_withLabelled'][g] * alpha[g] / (
            eo['eo_bias_withLabelled'][g] * alpha[g] + pe['pe_bias_withLabelled'][g] * (1 - alpha[g]))
                             for g in groups}
    ppv_group_difference = {'unbiased': ppv_ub[0] - ppv_ub[1],
                            'corrected_l': ppv_bias_withLabelled[0] - ppv_bias_withLabelled[1]}
    return ppv_group_difference



def TPR(Y, YPred, thr=0.5, weights=None):
    Y = np.squeeze(Y)
    if weights is None:
        weights = np.ones_like(Y)
    else:
        weights = np.reshape(weights, Y.shape)
    # return np.sum(weights * Y * (np.reshape(YPred, Y.shape) >= thr)) / np.sum(weights * Y)
    return np.sum(weights * Y * (np.reshape(YPred, Y.shape) >= thr)) / np.sum(Y)


def FPR(Y, YPred, thr=0.5, weights=None):
    Y = np.squeeze(Y)
    if weights is None:
        weights = np.ones_like(Y)
    else:
        weights = np.reshape(weights, Y.shape)
    # return np.sum(weights * (1-Y) * (np.reshape(YPred, Y.shape) >= thr)) / np.sum(weights * (1-Y))
    return np.sum(weights * (1 - Y) * (np.reshape(YPred, Y.shape) >= thr)) / np.sum((1 - Y))
