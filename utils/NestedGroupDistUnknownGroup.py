import time
import copy

import numpy as np
from scipy.linalg import sqrtm
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans

from utils.emutils import em_opt
from utils.NestedGroupDist import NestedGroupDist

class NestedGroupDistUnknownGroup(NestedGroupDist):

    def __init__(self, x_unlabeled=None,
                 x_labeled=None,
                 unlabeled_groups=None,
                 components=None, num_classes: int = 2, num_groups: int = 2):
        super(NestedGroupDistUnknownGroup, self).__init__(
            x_unlabeled=x_unlabeled, x_labeled=x_labeled,
            unlabeled_groups=unlabeled_groups, labeled_groups=None,
            components=components, num_classes=num_classes, num_groups=num_groups)
        """
        Use this class when labeled data has unknown group information
        :param x_unlabeled: numpy array of points shape=[num_points,dim]
        :param x_labeled: iterable of numpy arrays, corresponding to each class.
                          The index matches that of Ks, e.g. the first array here corresponds to the class
                          listed first in Ks
        :param unlabeled_groups: group labels for X_unlabeled
        :param components: iterable of number of components or a single integer. If a single integer, the same
                   number of components for each class will be used
        :param num_classes: int indicating the number of classes
        """

        self.X_unlabeled = x_unlabeled
        self.X_labeled = x_labeled
        self.num_classes = num_classes
        self.unlabeled_groups = unlabeled_groups
        self.labeled_groups = None
        self.num_groups = num_groups

        assert len(self.X_labeled) == self.num_classes
        assert len(np.unique(self.unlabeled_groups)) == self.num_groups
        # all groups represented in unlabeled data:
        # TODO: do all groups need to be represented in each class of labeled data?

        # Number of components per class
        if len(components) == 1:
            self.Ks = components * np.ones(self.num_classes)
        else:
            self.Ks = components

        self.dim = self.X_unlabeled.shape[1]

        self.alphas = np.empty([self.num_groups, self.num_classes])  #
        self.w = [np.empty([self.num_groups, self.Ks[c]]) for c in range(self.num_classes)]
        self.w_labeled = [np.empty(self.Ks[c]) for c in range(self.num_classes)]
        self.sg = None
        self.mu = None

        # state of actively optimizing
        self.training = False
        self._set_convergence_cond()  # initialize optimization parameters
        self.step = 0
        self.lls = []

    @staticmethod
    def get_group_points(points, indices, group):
        return points[indices == group]

    def opt_step(self):
        """Increase optimization step by one"""
        self.step += 1

    def estimate_params(self, max_steps: int = 2000, class_scales=None,
                        rnd_state: int = 0, initialization: str = 'labeled_means'):
        """
        :param max_steps: Maximum number of steps in EM optimization
        :param class_scales: numpy array of scaling weights for each class to use during
                             optimization. If None, weights of 1 is used for every class
        :param rnd_state: random state for initialization
        :param initialization: 'labeled_means' to initialize means from labeled means
        """
        self._set_convergence_cond()
        self.converg['max_steps'] = max_steps
        self.step = 0
        self.training = True

        if class_scales is None:
            class_scales = np.ones(self.num_classes)  # factor by which to scale class in labeled data

        n_unlabeled = self.X_unlabeled.shape[0]
        n_unlabeled_pergroup = [(self.unlabeled_groups==g).sum().item() for g in range(self.num_groups)]
        n_labeled = [self.X_labeled[c].shape[0] for c in range(self.num_classes)]  # num labeled samples per class
        alpha_l = np.array(n_labeled)/sum(n_labeled)


        # initialize unlabeled means with kmeans
        kmeans = KMeans(n_clusters=np.sum(self.Ks), random_state=rnd_state, init='k-means++').fit(self.X_unlabeled)
        k_mu = kmeans.cluster_centers_  # k_mu should be shaped 2K*dim

        # Initialize mu and w_labeled based only on labeled data
        self._initialize_from_labeled(k_mu, initialization, rnd_state)

        # update alphas, w, and sg
        self._initialize_params_from_mu(self.mu)

        for c in range(self.num_classes):
            self.sg[c] = self._check_conditioning(self.sg[c], self.Ks[c])

        # Start Optimization
        start_time = time.time()
        ll, _ = self.pnu_loglikelihood(class_scales)

        self.lls.append(ll)

        # Iterate optimization
        while self.training:
            mu_old = copy.deepcopy(self.mu)

            # E step
            # terms from unlabeled points
            posteriors = self.compute_posteriors(self.X_unlabeled, self.unlabeled_groups,
                                                 self.w, list(range(self.num_classes)), self.alphas)

            # terms from labeled points
            posteriors_labeled = [np.zeros([n_labeled[c], self.Ks[c]]) for c in range(self.num_classes)]
            for c in range(self.num_classes):
                num_in_class = n_labeled[c]
                if num_in_class > 0:
                    posteriors_labeled[c] = self.compute_posteriors_nogroups(
                        self.X_labeled[c], self.w_labeled, [c], alpha_l)[0]

            # M step
            # given label and component posteriors, update parameters: alphas, w, mu, and sg
            # alphas and ws are estimated using only unlabeled data

            # Update parameters alpha and w (depend only on unlabeled data) and w_labeled (labeled data)
            # use sorted sums for more accurate result
            p_g = [np.zeros([self.num_groups, self.Ks[c]]) for c in range(self.num_classes)]
            p_labeled = [np.zeros([self.Ks[c]]) for c in range(self.num_classes)]
            for c in range(self.num_classes):
                for k in range(self.Ks[c]):
                    for g in range(self.num_groups):
                        p_g[c][g, k] = np.sum(posteriors[c][self.unlabeled_groups == g, k])
                        if n_labeled[c] > 0:
                            p_labeled[c][k] = class_scales[c] * np.sum(posteriors_labeled[c][:, k])

            for c in range(self.num_classes):
                for g in range(self.num_groups):
                    comp_posterior_sum = np.sum(p_g[c][g, :])  # sum over subcomponents
                    self.alphas[g][c] = comp_posterior_sum / n_unlabeled_pergroup[g]
                    for k in range(self.Ks[c]):
                        self.w[c][g][k] = np.sum(p_g[c][g, k]) / comp_posterior_sum
                        if n_labeled[c] > 0:
                            lcomp_posterior_sum = np.sum(p_labeled[c])
                            self.w_labeled[c][k] = p_labeled[c][k] / lcomp_posterior_sum

            # Correct mixing proportions
            # prevent later taking log(w_i) if w_i==0
            for c in range(self.num_classes):
                for g in range(self.num_groups):
                    if np.sum(self.w[c][g] == 0) > 0:
                        self.w[c][g][self.w[c][g] == 0] = self.converg['eps']
                    self.w[c][g] = self.w[c][g] / np.sum(self.w[c][g])

                if n_labeled[c] > 0:
                    self.w_labeled[c][self.w_labeled[c] == 0] = self.converg['eps']
                    self.w_labeled[c] = self.w_labeled[c] / sum(self.w_labeled[c])

            # Update parameters mu & sigma
            # sum posteriors over the points
            denom = [np.sum(posteriors[c], axis=0) for c in range(self.num_classes)]
            for c in range(self.num_classes):
                if n_labeled[c] > 0:
                    denom[c] += p_labeled[c]

            self.mu = [np.zeros([self.dim, self.Ks[c]]) for c in range(self.num_classes)]
            self.sg = [np.zeros([self.dim, self.dim, self.Ks[c]]) for c in range(self.num_classes)]

            for c in range(self.num_classes):
                for k in range(self.Ks[c]):
                    pX = posteriors[c][:, k][:, np.newaxis] * self.X_unlabeled
                    # mu(c, :, k) = np.sum(sort(pX))
                    self.mu[c][:, k] = np.sum(pX, axis=0)
                    xmu_unlabeled = self.X_unlabeled - mu_old[c][:, k]
                    pxmu = np.sqrt(posteriors[c][:, k])[:, np.newaxis] * xmu_unlabeled
                    self.sg[c][:, :, k] = np.reshape(np.matmul(np.transpose(pxmu), pxmu),
                                                     [self.dim, self.dim])

                    if n_labeled[c] > 0:
                        xmu_labeled = self.X_labeled[c] - mu_old[c][:, k]
                        pX = posteriors_labeled[c][:, k][:, np.newaxis] * self.X_labeled[c]
                        self.mu[c][:, k] = self.mu[c][:, k] + (class_scales[c] * np.sum(pX, axis=0))
                        pxmu = np.sqrt(posteriors_labeled[c][:, k])[:, np.newaxis] * xmu_labeled
                        self.sg[c][:, :, k] = self.sg[c][:, :, k] + (
                                class_scales[c] * np.reshape(np.matmul(np.transpose(pxmu), pxmu), [self.dim, self.dim]))

            for c in range(self.num_classes):
                denom[c][denom[c] == 0] = self.converg['eps']
                for k in range(self.Ks[c]):
                    self.mu[c][:, k] = self.mu[c][:, k] / denom[c][k]
                    self.sg[c][:, :, k] = self.sg[c][:, :, k] / denom[c][k]

            # recondition covariance matrix if necessary
            for c in range(self.num_classes):
                self.sg[c] = self._check_conditioning(self.sg[c], self.Ks[c])

            # Compute loglikelihood
            ll, _ = self.pnu_loglikelihood(class_scales)
            self.lls.append(ll)

            self.opt_step()  # iterate step count

            # Check if termatination conditions
            self._check_termination(mu_old)

        # reshape mu to num_components * dim for each class
        mus = [self.mu[c].T for c in range(self.num_classes)]
        # reshape sigma to num_components * dim * dim for each class
        # during optimization, it is dim*dim*num_components
        covariance = [np.transpose(self.sg[c], axes=[2, 1, 0]) for c in range(self.num_classes)]

        self.mu = mus
        self.sg = covariance

        elapsed = time.time() - start_time
        # print(f'Parameters estimated in {elapsed:.2f} sec.')

    def _check_termination(self, mu_old):
        param_diff = np.sum([np.sum(np.abs(self.mu[c] - mu_old[c])) / np.abs(np.sum(self.mu[c]))
                             for c in range(self.num_classes)])
        ll_diff = np.abs(self.lls[-1] - self.lls[-2]) / np.abs(self.lls[-2])

        if ll_diff < self.converg['tol']:
            self.optim_counts['ll_diff_tol_count'] += 1
        else:
            self.optim_counts['ll_diff_tol_count'] = 0  # reset the count

        if param_diff < self.converg['param_tol']:
            self.optim_counts['param_diff_tol_count'] += 1
        else:
            self.optim_counts['param_diff_tol_count'] = 0

        if self.step >= self.converg['max_steps']:
            self.training = False
        ll_convergence = self.optim_counts['ll_diff_tol_count'] > self.converg['min_diff_steps']
        mu_convergence = self.optim_counts['param_diff_tol_count'] > self.converg['min_param_diff_steps']
        if ll_convergence and mu_convergence:
            self.training = False

    def _set_convergence_cond(self):
        # convergence conditions
        self.converg = dict(tol=1e-11, param_tol=1e-8, max_steps=1000,
                            eps=1e-300, min_diff_steps=100, min_param_diff_steps=10)
        self.optim_counts = dict(diff_tol_count=0, param_diff_tol_count=0)

    def _check_conditioning(self, sig, K):
        sig = copy.deepcopy(sig)
        cond_min = 1000
        eps = 0.001
        if len(sig.shape) == 4:
            for c in range(2):
                for k in range(K):
                    s_cond = np.linalg.cond(sig[c, :, :, k])
                    if s_cond > cond_min:
                        sig[c, :, :, k] = self._recondition_sig(sig[c, :, :, k], cond_min, self.dim, eps)
        elif len(sig.shape) == 3:
            for k in range(K):
                s_cond = np.linalg.cond(sig[:, :, k])
                if s_cond > cond_min:
                    sig[:, :, k] = self._recondition_sig(sig[:, :, k], cond_min, self.dim, eps)
        return sig

    @staticmethod
    def _recondition_sig(sig_orig, cond_min, dim, eps):
        sig = sig_orig + (eps * np.eye(dim))
        s_cond = np.linalg.cond(sig)
        while s_cond > cond_min:
            eps = eps * 2
            sig = sig_orig + (eps * np.eye(dim))
            s_cond = np.linalg.cond(sig)
        return sig

    def _init_empty_params(self):
        self.sg = [np.zeros([self.dim, self.dim, self.Ks[c]]) for c in range(self.num_classes)]
        for c in range(self.num_classes):
            for k in range(self.Ks[c]):
                self.sg[c][:, :, k] = np.eye(self.dim)

        self.mu = [np.zeros([self.dim, self.Ks[c]]) for c in range(self.num_classes)]
        self.w_labeled = [np.ones([self.num_groups, self.Ks[c]]) / self.Ks[c]
                          for c in range(self.num_classes)]

    def _initialize_from_labeled(self, k_mu, initialization, rnd_state):
        # returns mu (shape= dim,dim,K), sig (shape=2,dim,dim,K), w_labeled (shape=2,K)

        # Todo: use the means provided by k_mu

        self._init_empty_params()

        # number of  labeled samples in class c
        n_labeled = np.array([self.X_labeled[c].shape[0] for c in range(self.num_classes)])

        if n_labeled.sum() == 0:
            # return initialized "empty" params if no labeled data
            return

        # We use all labeled data, ignoring group membership,
        # which means w_labeled will be initialized to the same values
        mu_labeled = [np.empty([self.Ks[c], self.dim]) for c in range(self.num_classes)]
        sg_labeled = [np.empty([self.Ks[c], self.dim, self.dim]) for c in range(self.num_classes)]
        for c in range(self.num_classes):
            if n_labeled[c].sum() > 0:
                w_labeled, mu_labeled[c], sg_labeled[c] = em_opt(
                    self.X_labeled[c], self.Ks[c], rnd_state=rnd_state)

                self.w_labeled[c] = w_labeled

        if initialization == 'labeled_means':  # use closest labeled means as unlabeled means
            for c in range(self.num_classes):
                if n_labeled[c].sum() > 0:
                    self.mu[c] = mu_labeled[c]
                    self.sg[c] = sg_labeled[c]

    def _initialize_params_from_mu(self, mu):
        """Calculate alphas, ws, and sigma for dataset with points X and means mu
        Points will be assigned to the closest mean mu(i,:) and used to compute
        parameters for each component."""

        Xgs = [self.get_group_points(self.X_unlabeled, self.unlabeled_groups, g)
               for g in range(self.num_groups)]

        # distance from points to each component means for each class
        dist_to_class_comps = [cdist(self.X_unlabeled, mu[c].transpose())
                               for c in range(self.num_classes)]
        # distance to the closest component in each class
        dist_to_c = np.vstack([dist_to_class_comps[c].min(1) for c in range(self.num_classes)])
        label_c_comp = [dist_to_class_comps[c].argmin(1) for c in range(self.num_classes)]

        # assign to the closest class
        label_c = dist_to_c.argmin(0)
        X_c = [[self.X_unlabeled[np.logical_and((label_c == c), (self.unlabeled_groups == g))]
                for g in range(self.num_groups)]
               for c in range(self.num_classes)]

        num_c = [[X_c[c][g].shape[0] for g in range(self.num_groups)]
                 for c in range(self.num_classes)]

        # initialize alpha based on assignment counts
        self.alphas = np.array(num_c).transpose() / np.array([x.shape[0] for x in Xgs])

        # Initialize weights and covariance
        self.w = [[np.ones([self.Ks[c]]) / self.Ks[c] for g in range(self.num_groups)]
                  for c in range(self.num_classes)]
        self.sg = [np.zeros([self.dim, self.dim, self.Ks[c]]) for c in range(self.num_classes)]
        for c in range(self.num_classes):
            for k in range(self.Ks[c]):
                self.sg[c][:, :, k] = np.eye(self.dim)

        for c in range(self.num_classes):
            for k in range(self.Ks[c]):
                c_component = label_c_comp[c]  # components assignment

                if np.sum(c_component == k) > 0:  # at least some points assigned to this component in this class
                    in_component = label_c_comp[c] == k
                    in_class = label_c == c

                    t = self.X_unlabeled[np.logical_and(in_component, in_class), :] - mu[c][:, k]
                    self.sg[c][:, :, k] = np.matmul(np.transpose(t), t) / np.sum(c_component == k)

                    # weights are group-specific
                    for g in range(self.num_groups):
                        in_group = self.unlabeled_groups == g
                        self.w[c][g][k] = np.sum(np.logical_and(in_component, in_group)) / num_c[c][g]

        # normalize weights
        self.w = [[self.w[c][g] / sum(self.w[c][g]) for g in range(self.num_groups)]
                  for c in range(self.num_classes)]

    def compute_ll(self, X, group_membership, mu, sigma, w, Ks, num_classes):
        # mu should be dim*number of components
        # sigma should be dim*dim*number of components

        N_g = np.zeros(self.num_groups).astype(int)
        for g in range(self.num_groups):
            N_g[g] = self.get_group_points(X, group_membership, g).shape[0]

        ll = [[np.zeros([N_g[g]]) for g in range(self.num_groups)] for _ in range(num_classes)]

        twopidim = (2 * np.pi) ** self.dim

        for c in range(num_classes):
            if num_classes > 1:
                m = mu[c]  # reshape to [dim,K]?
                sg = sigma[c]  # reshape to [dim,dim,K]?
            else:
                m = mu
                sg = sigma

            for g in range(self.num_groups):
                l = np.zeros([N_g[g], Ks[c]])
                Xg = self.get_group_points(X, group_membership, g)
                for k in range(Ks[c]):
                    sig_ij = sg[:, :, k]  # select sigma for component c
                    detsig = np.sqrt(twopidim * np.linalg.det(sig_ij))
                    xdiff = Xg - m[:, k]
                    squareterm = np.sum(np.matmul(xdiff, np.linalg.inv(sig_ij)) * xdiff, axis=1)
                    #             squareterm = np.sum((xdiff / sig_ij) * xdiff,axis=1)  # equivalend to (xdiff * inv(sig_ij)) .* xdiff
                    if num_classes > 1:
                        wk = w[c][g][k]
                    else:
                        wk = w[g][k]

                    N_ck = wk * np.exp(-0.5 * squareterm) / detsig
                    N_ck[N_ck == 0] = self.converg['eps']
                    l[:, k] = N_ck
                ll[c][g] = np.sum(l, axis=1)  # sums over component terms

        loglikelihood = [None for _ in range(self.num_groups)]
        for g in range(self.num_groups):
            per_point_ll = np.vstack([ll[c][g] for c in range(num_classes)]).T.sum(axis=1)
            loglikelihood[g] = np.sum(np.log(per_point_ll))
        return sum(loglikelihood)


    def compute_ll_nogroup(self, X, mu, sigma, w, Ks, num_classes):
        # mu should be dim*number of components
        # sigma should be dim*dim*number of components

        N = X.shape[0]

        ll = [np.zeros(N) for _ in range(num_classes)]

        twopidim = (2 * np.pi) ** self.dim

        for c in range(num_classes):
            if num_classes > 1:
                m = mu[c]  # reshape to [dim,K]?
                sg = sigma[c]  # reshape to [dim,dim,K]?
            else:
                m = mu
                sg = sigma

            l = np.zeros([N, Ks[c]])
            for k in range(Ks[c]):
                sig_ij = sg[:, :, k]  # select sigma for component c
                detsig = np.sqrt(twopidim * np.linalg.det(sig_ij))
                xdiff = X - m[:, k]
                squareterm = np.sum(np.matmul(xdiff, np.linalg.inv(sig_ij)) * xdiff, axis=1)
                #             squareterm = np.sum((xdiff / sig_ij) * xdiff,axis=1)  # equivalend to (xdiff * inv(sig_ij)) .* xdiff
                if num_classes > 1:
                    wk = w[c][k]
                else:
                    wk = w[k]

                N_ck = wk * np.exp(-0.5 * squareterm) / detsig
                N_ck[N_ck == 0] = self.converg['eps']
                l[:, k] = N_ck
            ll[c] = np.sum(l, axis=1)  # sums over component terms

        per_point_ll = np.vstack([ll[c] for c in range(num_classes)]).T.sum(axis=1)
        loglikelihood = np.sum(np.log(per_point_ll))
        return loglikelihood

    def pnu_loglikelihood(self, class_scales):

        # Compute loglikelihood for all components (unlabeled, labeled for each class c)
        N_unlabeled = self.X_unlabeled.shape[0]  # num unlabeled samples
        N_labeled = [self.X_labeled[c].shape[0] for c in range(self.num_classes)]  # num labeled samples in each class

        # Unlabeled contribution
        w_ = [[self.alphas[g][c] * self.w[c][g] for g in range(self.num_groups)]
              for c in range(self.num_classes)]  # todo: check this scaling
        ll = self.compute_ll(self.X_unlabeled, self.unlabeled_groups,
                             self.mu, self.sg, w_, self.Ks,
                             num_classes=self.num_classes)

        ll_labeled = [0 for _ in range(self.num_classes)]
        ll_l = [0 for _ in range(self.num_classes)]
        for c in range(self.num_classes):
            if N_labeled[c] > 0:
                ll_labeled[c] = self.compute_ll_nogroup(self.X_labeled[c],
                                                        self.mu[c], self.sg[c],
                                                        self.w_labeled[c], [self.Ks[c]],
                                                        num_classes=1)
                ll_l[c] = ll_labeled[c] / N_labeled[c]

        ll_unlabeled = ll / N_unlabeled
        for c in range(self.num_classes):
            if N_labeled[c] > 0:
                ll += class_scales[c] * ll_labeled[c]

        logl = ll / (N_unlabeled + sum([class_scales[c] * N_labeled[c]]))

        return logl, ll_unlabeled

    def compute_posteriors(self, X, group_labels, weights, classes, alphas):

        n = X.shape[0]
        posteriors = [np.zeros([n, self.Ks[c]]) for c in range(self.num_classes)]
        numerator_N = [np.zeros([n, self.Ks[c]]) for c in range(self.num_classes)]

        for c in classes:
            inv_sig = np.zeros([self.Ks[c], self.dim, self.dim])
            det_sig = np.zeros([self.Ks[c]])
            mult = np.zeros([self.Ks[c], self.num_groups])  # assumes all groups are present in each class
            for k in range(self.Ks[c]):  # components within class

                sig = np.reshape(self.sg[c][:, :, k],
                                 [self.dim, self.dim])  # select sigma for component c, subcomponent k
                inv_sig[k, :, :] = np.linalg.inv(sig)
                sqrtinvsig = sqrtm(inv_sig[k, :, :])
                det_sig[k] = np.linalg.det(sig)

                for g in range(self.num_groups):
                    t_N = X[group_labels == g, :] - self.mu[c][:, k]  # N x dim
                    expsum = np.sum(np.matmul(t_N, sqrtinvsig) ** 2, axis=1)
                    mult[k, g] = alphas[g][c] * weights[c][g][k] / np.sqrt((2 * np.pi) ** self.dim * det_sig[k])

                    numerator_N[c][group_labels == g, k] = mult[k,g] * np.exp(-0.5 * expsum)

        # denominator is sum over classes and components
        denom = np.vstack([np.sum(numerator_N[c], axis=1) for c in classes]).sum(axis=0)
        denom[denom == 0] = self.converg['eps']
        for c in classes:
            for k in range(self.Ks[c]):
                posteriors[c][:, k] = numerator_N[c][:, k] / denom

        return [posteriors[c] for c in classes]

    def compute_posteriors_nogroups(self, X, weights, classes, alphas):

        n = X.shape[0]
        posteriors = [np.zeros([n, self.Ks[c]]) for c in range(self.num_classes)]
        numerator_N = [np.zeros([n, self.Ks[c]]) for c in range(self.num_classes)]

        for c in classes:
            inv_sig = np.zeros([self.Ks[c], self.dim, self.dim])
            det_sig = np.zeros([self.Ks[c]])
            mult = np.zeros([self.Ks[c]])
            for k in range(self.Ks[c]):  # components within class

                sig = np.reshape(self.sg[c][:, :, k],
                                 [self.dim, self.dim])  # select sigma for component c, subcomponent k
                inv_sig[k, :, :] = np.linalg.inv(sig)
                sqrtinvsig = sqrtm(inv_sig[k, :, :])
                det_sig[k] = np.linalg.det(sig)

                t_N = X - self.mu[c][:, k]  # N x dim
                expsum = np.sum(np.matmul(t_N, sqrtinvsig) ** 2, axis=1)
                mult[k] = alphas[c] * weights[c][k] / np.sqrt((2 * np.pi) ** self.dim * det_sig[k])

                numerator_N[c][:, k] = mult[k] * np.exp(-0.5 * expsum)

        # denominator is sum over classes and components
        denom = np.vstack([np.sum(numerator_N[c], axis=1) for c in classes]).sum(axis=0)
        denom[denom == 0] = self.converg['eps']
        for c in classes:
            for k in range(self.Ks[c]):
                posteriors[c][:, k] = numerator_N[c][:, k] / denom

        return [posteriors[c] for c in classes]