import numpy as np


def correct_dp(points_bias, y_all_bias, y_all_ub, p, q):
    # Demographic parity
    dp_bias = {}
    dp_ub = {}
    dp_bias_uncorrected = {}
    for group in points_bias.keys():
        pdf_p = p[group].points_pdf(points_bias[group])
        pdf_q = q[group].points_pdf(points_bias[group])

        dp_bias[group] = sum(y_all_bias[group] * (pdf_p / pdf_q)) / y_all_bias[group].shape[0]
        dp_ub[group] = sum(y_all_ub[group]) / y_all_ub[group].shape[0]
        dp_bias_uncorrected[group] = sum(y_all_bias[group]) / y_all_bias[group].shape[0]

    dp_group_difference = {'unbiased': dp_ub[0] - dp_ub[1],
                           'uncorrected': dp_bias_uncorrected[0] - dp_bias_uncorrected[1],
                           'corrected': dp_bias[0] - dp_bias[1]}
    return dp_group_difference, {'dp_bias': dp_bias, 'dp_ub': dp_ub, 'dp_bias_uncorrected': dp_bias_uncorrected}


def correct_eo(points_pos_bias, y_pos_bias, y_pos_ub, d_pos_bias, d_pos_ub):
    # Equal opportunity
    eo_bias = {}
    eo_ub = {}
    eo_bias_uncorrected = {}
    for group in points_pos_bias.keys():
        pdf_unbiased = d_pos_ub[group].points_pdf(points_pos_bias[group])
        pdf_biased = d_pos_bias[group].points_pdf(points_pos_bias[group])

        eo_bias[group] = sum(y_pos_bias[group] * pdf_unbiased/pdf_biased) / y_pos_bias[group].shape[0]
        eo_ub[group] = sum(y_pos_ub[group]) / y_pos_ub[group].shape[0]
        eo_bias_uncorrected[group] = sum(y_pos_bias[group]) / y_pos_bias[group].shape[0]

    eo_group_difference = {'unbiased': eo_ub[0] - eo_ub[1],
                           'uncorrected': eo_bias_uncorrected[0] - eo_bias_uncorrected[1],
                           'corrected': eo_bias[0] - eo_bias[1]}
    return eo_group_difference, {'eo_bias': eo_bias, 'eo_ub': eo_ub, 'eo_bias_uncorrected': eo_bias_uncorrected}


def correct_pe(points_neg_bias, y_neg_bias, y_neg_ub, d_neg_bias, d_neg_ub):
    # predictive equality
    pe_bias = {}
    pe_ub = {}
    pe_bias_uncorrected = {}
    for group in points_neg_bias.keys():
        pdf_ub = d_neg_ub[group].points_pdf(points_neg_bias[group])
        pdf_b = d_neg_bias[group].points_pdf(points_neg_bias[group])

        pe_bias[group] = sum(y_neg_bias[group] * pdf_ub / pdf_b) / y_neg_bias[group].shape[0]
        pe_ub[group] = sum(y_neg_ub[group]) / y_neg_ub[group].shape[0]
        pe_bias_uncorrected[group] = sum(y_neg_bias[group]) / y_neg_bias[group].shape[0]

    pe_group_difference = {'unbiased': pe_ub[0] - pe_ub[1],
                           'uncorrected': pe_bias_uncorrected[0] - pe_bias_uncorrected[1],
                           'corrected': pe_bias[0] - pe_bias[1]}
    return pe_group_difference, {'pe_bias': pe_bias, 'pe_ub': pe_ub, 'pe_bias_uncorrected': pe_bias_uncorrected}


def correct_ppv(eo, pe, alpha, alpha_bias):
    # predictive positive value = TPR*alpha / (TPR*alpha + FPR*(1-alpha))
    ppv_bias = {}
    ppv_ub = {}
    ppv_bias_uncorrected = {}
    for group in eo['eo_bias'].keys():
        ppv_bias[group] = eo['eo_bias'][group] * alpha[group] / (eo['eo_bias'][group] * alpha[group]
                                                  + pe['pe_bias'][group] * (1 - alpha[group]))
        ppv_ub[group] = eo['eo_ub'][group] * alpha[group] / (eo['eo_ub'][group] * alpha[group]
                                                             + pe['pe_ub'][group] * (1 - alpha[group]))
        ppv_bias_uncorrected[group] = eo['eo_bias_uncorrected'][group] * alpha_bias[group] / (
            eo['eo_bias_uncorrected'][group] * alpha_bias[group] + pe['pe_bias_uncorrected'][group] * (1 - alpha_bias[group]))

    ppv_group_difference = {'unbiased': ppv_ub[0] - ppv_ub[1],
                            'uncorrected': ppv_bias_uncorrected[0] - ppv_bias_uncorrected[1],
                            'corrected': ppv_bias[0] - ppv_bias[1]}
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
