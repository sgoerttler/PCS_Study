import os
import numpy as np
from scipy.optimize import curve_fit
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

from polynomial_regression_model import RegressionModel
from utils import normalize_data, transparent_to_opaque


class EffectSizeSimulator(object):
    """ Class for simulating effect sizes in the PCS experiment.

    Parameters
    ----------
    dir_save : str
        Directory for saving simulation results
    verbose : bool
        Print progress information

    Attributes
    ----------
    dir_save : str
        Directory for saving simulation results
    verbose : bool
        Print progress information
    num_trials : int
        Number of trials per participant
    num_colors : int
        Number of colors
    num_conds : int
        Number of conditions
    RT_mean : float
        Mean response time
    RT_std : float
        Standard deviation of response time
    RT_max : float
        Maximum response time
    part_mean_std : float
        Standard deviation of participant mean effect
    part_std_std : float
        Standard deviation of participant standard deviation
    scale : float
        Scale parameter for psychometric function
    mean_diff_35 : float
        Mean difference for 35% accuracy
    mean_diff_75 : float
        Mean difference for 75% accuracy
    """

    def __init__(self, dir_save, verbose=True):
        # save directory
        self.dir_save = dir_save
        os.makedirs(dir_save, exist_ok=True)
        self.verbose=verbose

        # experiment parameters
        self.num_trials = 1144
        self.num_colors = 4
        self.num_conds = 2

        # RT priors (based on literature)
        self.RT_mean = 600
        self.RT_std = 200
        self.RT_max = 1000
        self.part_mean_std = 100
        self.part_std_std = 10

        # estimate sensible scale for psychometric function
        self.scale = 1 * np.abs(self.inv_psychometric_curve(0.75) - self.inv_psychometric_curve(0.35))

        cdf_psycho_calibration = self.get_cdf_psycho_calibration(self.scale)
        self.mean_diff_35 = self.inv_psychometric_curve(0.35) / cdf_psycho_calibration
        self.mean_diff_75 = self.inv_psychometric_curve(0.75) / cdf_psycho_calibration

    def simulate_single_participant(self, effect_size_1, effect_size_2):
        part_rand_eff_RT_mean = np.random.normal(0, self.part_mean_std)
        part_rand_eff_RT_std = np.random.normal(0, self.part_std_std)

        # simulate independent variables in experiment
        colors = np.repeat(np.arange(self.num_colors), self.num_trials // self.num_colors)
        stim_diff_35 = self.psychometric_curve(np.random.normal(size=self.num_trials // 2), x0=self.mean_diff_35, k=self.scale)
        stim_diff_75 = self.psychometric_curve(np.random.normal(size=self.num_trials // 2), x0=self.mean_diff_75, k=self.scale)
        stim_diff = np.zeros(self.num_trials)
        stim_diff[:self.num_trials // self.num_conds] = np.random.permutation(stim_diff_35)
        stim_diff[self.num_trials // self.num_conds:] = np.random.permutation(stim_diff_75)
        acc_ind = (np.random.uniform(0, 1, self.num_trials) < stim_diff).astype(int)
        data_paired_part = {
            'target_acc_75': np.repeat(np.arange(self.num_conds), self.num_trials // self.num_conds),
            'green': (colors == 1).astype(int),
            'blue': (colors == 2).astype(int),
            'yellow': (colors == 3).astype(int),
            'stim_acc_ded': stim_diff,
            'stim_acc_ded_sq_scaled': stim_diff ** 2 * 0.01,
            'stim_acc_ded_cu_scaled': stim_diff ** 2 * 0.0001,
            'neg_acc_ind': 1 - acc_ind
        }

        # simulate dependent variable RT according to model
        RT_mean = self.RT_mean + part_rand_eff_RT_mean
        RT_std = self.RT_std + part_rand_eff_RT_std
        mu = np.log(RT_mean ** 2 / np.sqrt(RT_std ** 2 + RT_mean ** 2))  # Convert mean/std to log-normal params
        sigma = np.sqrt(np.log(1 + (RT_std ** 2 / RT_mean ** 2)))
        data_paired_part['RT_ded'] = np.random.lognormal(mu, sigma, self.num_trials)

        data_paired_part = self.remove_slow_trials(data_paired_part)

        # add effects
        data_paired_part['RT_ded'] += normalize_data(self.stim_diff_to_RT_curve(data_paired_part['stim_acc_ded'])) * self.RT_std * effect_size_1
        data_paired_part['RT_ded'] += normalize_data(data_paired_part['stim_acc_ded'] * data_paired_part['neg_acc_ind']) * self.RT_std * effect_size_2

        return data_paired_part

    def get_exp_effect_size(self, n_participants, effect_size_1, effect_size_2):
        for part in range(n_participants):
            data_paired_part = self.simulate_single_participant(effect_size_1, effect_size_2)

            if part == 0:
                data_paired = data_paired_part
            else:
                for key in data_paired.keys():
                    data_paired[key] = np.concatenate((data_paired[key], data_paired_part[key]))

        model_sim = smf.ols(RegressionModel.get_formula_extended(), data_paired).fit()
        model_sim.f2 = RegressionModel.compute_cohens_f2_formula(RegressionModel.get_formula_extended(), data_paired, model_sim.rsquared)
        return model_sim.f2['stim_acc_ded'], model_sim.f2['neg_acc_ind:stim_acc_ded']

    def remove_slow_trials(self, data_dict):
        mask_slow = data_dict['RT_ded'] > self.RT_max
        for key in data_dict.keys():
            data_dict[key] = data_dict[key][~mask_slow]
        return data_dict

    def get_cdf_psycho_calibration(self, scale):
        try:
            return np.load(os.path.join(self.dir_save, 'cdf_psycho_calibration.npy'))
        except FileNotFoundError:
            x = np.linspace(-8, 8, 100)
            y = np.zeros(x.shape)
            for idx_x_i, x_i in enumerate(x):
                y[idx_x_i] = np.mean(self.psychometric_curve(np.random.normal(size=10_000), x0=x_i, k=scale))
            func = lambda x, k: self.psychometric_curve(x, k)

            # fitting
            popt, _ = curve_fit(func, x, y, p0=1)

            np.save(os.path.join(self.dir_save, 'cdf_psycho_calibration.npy'), popt[0])
            if self.verbose:
                print('CDF of psychometric curve calibrated!')
            return popt[0]

    def get_f2(self, effect_sizes, nums_participants, num_repeats=10):
        try:
            return self.load_results()
        except FileNotFoundError:
            return self.retrieve_simulated_f2(effect_sizes, nums_participants, num_repeats=num_repeats)

    def retrieve_simulated_f2(self, effect_sizes, nums_participants, num_repeats=10):
        f2_stim_acc_ded = np.zeros((len(effect_sizes), len(nums_participants)))
        f2_stim_acc_ded_interaction_neg_acc_ind = np.zeros((len(effect_sizes), len(nums_participants)))
        for idx_effect_size, effect_size in enumerate(effect_sizes):
            if self.verbose:
                print(f'effect size simulation: {idx_effect_size + 1} / {len(effect_sizes)}')

            for idx_num_participants, num_participants in enumerate(nums_participants):
                if self.verbose:
                    print(f'participant size: {idx_num_participants + 1} / {len(nums_participants)}')
                f2_var_1_i = []
                f2_var_2_i = []
                for repeats in range(num_repeats):
                    sim_eff_1, sim_eff_2 = self.get_exp_effect_size(num_participants, effect_size, effect_size)
                    f2_var_1_i.append(sim_eff_1)
                    f2_var_2_i.append(sim_eff_2)
                f2_var_1_i = np.array(f2_var_1_i)
                f2_var_2_i = np.array(f2_var_2_i)
                f2_stim_acc_ded[idx_effect_size, idx_num_participants] = np.mean(f2_var_1_i[f2_var_1_i > 0])
                f2_stim_acc_ded_interaction_neg_acc_ind[idx_effect_size, idx_num_participants] = np.mean(f2_var_2_i[f2_var_2_i > 0])
            if self.verbose:
                print()

        os.makedirs(self.dir_save, exist_ok=True)

        # save as npz file
        np.save(os.path.join(self.dir_save, 'f2_stim_acc_ded.npy'), f2_stim_acc_ded)
        np.save(os.path.join(self.dir_save, 'f2_stim_acc_ded_interaction_neg_acc_ind.npy'), f2_stim_acc_ded_interaction_neg_acc_ind)

        return f2_stim_acc_ded, f2_stim_acc_ded_interaction_neg_acc_ind

    def load_results(self):
        f2_stim_acc_ded = np.load(os.path.join(self.dir_save, 'f2_stim_acc_ded.npy'))
        f2_stim_acc_ded_interaction_neg_acc_ind = np.load(os.path.join(self.dir_save, 'f2_stim_acc_ded_interaction_neg_acc_ind.npy'))
        return f2_stim_acc_ded, f2_stim_acc_ded_interaction_neg_acc_ind

    def psychometric_curve(self, x, gam_guess=0.25, lam_guess=0.05, x0=0, k=1):
        return self.sigmoid(x, x0=x0, k=k) * ((1 - lam_guess) - gam_guess) + gam_guess

    def inv_psychometric_curve(self, y, gam_guess=0.25, lam_guess=0.05, x0=0, k=1):
        return self.inv_sigmoid((y - gam_guess) / ((1 - lam_guess) - gam_guess), x0=x0, k=k)

    @staticmethod
    def stim_diff_to_RT_curve(stim_diff, gam_guess=0.25, lam_lapse=0.05):
        # inverse U-shape as proposed by literature
        return -(stim_diff - gam_guess - ((1 - lam_lapse) - gam_guess) / 2) ** 2

    @staticmethod
    def sigmoid(x, x0=0, k=1):
        return 1 / (1 + np.exp(-k * (x - x0)))

    @staticmethod
    def inv_sigmoid(y, x0=0, k=1):
        return -np.log(1 / y - 1) / k + x0


class PlotResults():
    """ Class for plotting effect size analysis results.

    Parameters
    ----------
    f2_stim_acc_ded : np.ndarray
        Effect sizes for the main effect
    f2_stim_acc_ded_interaction_neg_acc_ind : np.ndarray
        Effect sizes for the interaction effect
    nums_participants : np.ndarray
        Number of participants
    dir_figures : str
        Directory for saving figures

    Attributes
    ----------
    x : np.ndarray
        x-axis values
    results : list
        List of effect size results
    nums_participants : np.ndarray
        Number of participants
    dir_figures : str
        Directory for saving figures
    min_detect_effect : np.ndarray
        Minimum detectable effect sizes
    min_detect_cohenf2 : np.ndarray
        Minimum detectable Cohen's f^2
    popts_breakdown : np.ndarray
        Fitted parameters for effect size breakdown
    popts_min_detect : np.ndarray
        Fitted parameters for minimum detectable effect sizes
    num_parts_target : int
        Target number of participants
    """

    def __init__(self, f2_stim_acc_ded, f2_stim_acc_ded_interaction_neg_acc_ind, nums_participants, dir_figures=None):
        self.x = np.geomspace(1e-4, 0.5, 1000).astype(np.float64)
        self.results = [f2_stim_acc_ded, f2_stim_acc_ded_interaction_neg_acc_ind]
        self.nums_participants = nums_participants
        self.dir_figures = dir_figures

        self.min_detect_effect = np.zeros((len(self.results), len(self.nums_participants)))
        self.min_detect_cohenf2 = np.zeros((len(self.results), len(self.nums_participants)))
        self.popts_breakdown = np.zeros((len(self.results), len(self.nums_participants)), dtype=object)
        self.popts_min_detect = np.zeros((len(self.results), 2), dtype=object)
        self.num_parts_target = 16

    def fit_effect_size_breakdown(self):
        for idx_res, res in enumerate(self.results):
            p0 = [1e-4, 2, 10]
            for idx_num_parts, num_parts in enumerate(self.nums_participants):
                popt, _ = curve_fit(self.model_effect_size_breakdown_log, effect_sizes, np.log(res[:, idx_num_parts]), p0=p0,
                                    maxfev=100000)
                p0 = popt
                self.min_detect_effect[idx_res, idx_num_parts] = popt[2]
                self.min_detect_cohenf2[idx_res, idx_num_parts] = self.model_effect_size_breakdown(popt[2], *popt)
                self.popts_breakdown[idx_res, idx_num_parts] = popt

    def fit_min_detectable_effect_sizes(self):
        for idx_min_detect_eff, min_detect_eff in enumerate([self.min_detect_effect, self.min_detect_cohenf2]):
            p0 = [-1, 2]
            for idx_var in range(2):
                popt, _ = curve_fit(self.power_law, self.nums_participants, min_detect_eff[idx_var, :], p0=p0,
                                    maxfev=1000, sigma=1./np.sqrt(self.nums_participants))
                p0 = popt
                self.popts_min_detect[idx_var, idx_min_detect_eff] = popt

    def plot_all(self):
        self.fig, self.ax = plt.subplots(figsize=(8, 10), nrows=4, ncols=2, height_ratios=[1.5, 0.0, 1, 1])
        self.ax[1, 0].remove()
        self.ax[1, 1].remove()
        self.plot_model_breakdown(sub_figure=True)
        self.plot_effect_sizes(sub_figure=True)
        self.ax[0, 0].set_title('difficulty')
        self.ax[0, 1].set_title('interaction post-error$-$difficulty')
        self.show_plot(savefig_name='effect_size_analysis_all')

    def plot_model_breakdown(self, sub_figure=False):
        if not sub_figure:
            self.fig, self.ax = plt.subplots(figsize=(8.4, 6), nrows=1, ncols=2)
        self.plot_model_breakdown_data()
        self.set_frame_details()

        self.add_legend(['k', 'k', 'k', transparent_to_opaque('k', np.sqrt(10 / 100)), transparent_to_opaque('k', np.sqrt(37 / 100)), transparent_to_opaque('k', np.sqrt(100 / 100))],
                        ['-', '--', 'o', 'full', 'full', 'full'],
                        ['measured', 'fit', 'breakdown', '10 parts.', '37 parts.', '100 parts.'],
                        [self.ax[0], self.ax[0, 0]][sub_figure], loc='upper left')
        self.set_labels_breakdown()
        if not sub_figure:
            plt.show()

    def plot_model_breakdown_data(self):
        for idx_res, res in enumerate(self.results):
            for idx_num_parts, num_parts in enumerate(self.nums_participants):
                if not num_parts in [10, 37, 100]:
                    continue

                self.ax[0, idx_res].loglog(self.x, self.model_effect_size_breakdown(self.x, *self.popts_breakdown[idx_res, idx_num_parts]), color=['tab:blue', 'tab:orange'][idx_res],
                                        alpha=np.sqrt(num_parts / max(self.nums_participants)), ls='--')
                self.ax[0, idx_res].loglog(effect_sizes, res[:, idx_num_parts], color=['tab:blue', 'tab:orange'][idx_res],
                           alpha=np.sqrt(num_parts / max(self.nums_participants)))
    
                self.ax[0, idx_res].scatter(self.min_detect_effect[idx_res, idx_num_parts], self.min_detect_cohenf2[idx_res, idx_num_parts],
                            color=transparent_to_opaque(['tab:blue', 'tab:orange'][idx_res],
                                                        alpha=np.sqrt(num_parts / max(self.nums_participants))), zorder=10,
                            edgecolors=['tab:blue', 'tab:orange'][idx_res])

    def set_labels_breakdown(self):
        for idx_var in np.arange(2):
            self.ax[0, idx_var].set_xlabel('effect size $\sigma_{eff} / \sigma_{RT}$')
        self.ax[0, 0].set_ylabel("Cohen's $f^2$")

    def plot_effect_sizes(self, sub_figure=False):
        if not sub_figure:
            self.fig, self.ax = plt.subplots(figsize=(8.4, 8.4), nrows=2, ncols=2)
        self.plot_effect_sizes_data()

        self.set_frame_details()
        self.set_labels_effect_size()
        self.add_legend(['k', 'k'], ['-', '--'],
                        ['min detect. effect', 'fit'], [self.ax[0, 0], self.ax[1, 0]][sub_figure], loc='lower left')
        self.add_inset_titles(sub_figure)
        if not sub_figure:
            plt.show()

    def plot_effect_sizes_data(self):
        ax_offset = self.ax.shape[0] - 2
        for idx_min_detect_eff, min_detect_eff in enumerate([self.min_detect_effect, self.min_detect_cohenf2]):
            for idx_var in np.arange(2):
                self.ax[idx_min_detect_eff + ax_offset, idx_var].loglog(self.nums_participants, min_detect_eff[idx_var, :],
                                                            color=['tab:blue', 'tab:orange'][idx_var])
                self.ax[idx_min_detect_eff + ax_offset, idx_var].loglog(self.nums_participants,
                                                            self.power_law(self.nums_participants, *self.popts_min_detect[idx_var, idx_min_detect_eff]),
                                                            color=['tab:blue', 'tab:orange'][idx_var], linestyle='dashed')
                y_target_part = self.power_law(self.num_parts_target, *self.popts_min_detect[idx_var, idx_min_detect_eff])

                self.ax[idx_min_detect_eff + ax_offset, idx_var].hlines(y_target_part, 0, self.num_parts_target, color='gray', linestyles='dashed', linewidth=1)
                self.ax[idx_min_detect_eff + ax_offset, idx_var].vlines(self.num_parts_target, 1e-12, y_target_part, color='gray', linestyles='dashed', linewidth=1)


                self.ax[idx_min_detect_eff + ax_offset, idx_var].set_xlim([5, 100])
                ylims = self.ax[idx_min_detect_eff + ax_offset, idx_var].get_ylim()

                if idx_min_detect_eff == 0:
                    text = '$\sigma_{eff} / \sigma_{RT}$ = ' + f'{y_target_part:.1e}'.replace('e-0', r'$\times 10^{-') + r'}$'
                else:
                    text = '$f^2$ = ' + f'{y_target_part:.1e}'.replace('e-0', r'$\times 10^{-') + r'}$'
                self.ax[idx_min_detect_eff + ax_offset, idx_var].text(self.num_parts_target + 2, y_target_part * (ylims[1] / ylims[0]) ** (1/8), f'$N_{{part}}$ = {self.num_parts_target}', ha='left')
                self.ax[idx_min_detect_eff + ax_offset, idx_var].text(self.num_parts_target + 2, y_target_part * (ylims[1] / ylims[0]) ** (1/30), text, ha='left')
                self.ax[idx_min_detect_eff + ax_offset, idx_var].scatter(self.num_parts_target, y_target_part, marker='o', color='gray', zorder=10, s=25)

    def set_labels_effect_size(self):
        ax_offset = self.ax.shape[0] - 2
        for idx_var in np.arange(2):
            self.ax[1 + ax_offset, idx_var].set_xlabel('sample size $N_{part}$ [#]')
        self.ax[0 + ax_offset, 0].set_ylabel('min. detect. effect size $\sigma_{eff} / \sigma_{RT}$')
        self.ax[1 + ax_offset, 0].set_ylabel("min. detect. Cohen's $f^2$")

        self.ax[0 + ax_offset, 0].set_yticks([3e-3, 4e-3, 6e-3, 1e-2], ['', '', '', r'$10^{-2}$'])

    def set_frame_details(self):
        for ax in self.ax.flatten():
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)

    def add_inset_titles(self, sub_figure):
        if self.ax.shape[0] == 2:
            idcs_axs = [(0, 0), (0, 1), (1, 0), (1, 1)]
            letters = ['A', 'B', 'C', 'D']
        elif self.ax.shape[0] == 3:
            idcs_axs = [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)]
            letters = ['A', 'B', 'C', 'D', 'E', 'F']
        else:
            idcs_axs = [(0, 0), (0, 1), (2, 0), (2, 1), (3, 0), (3, 1)]
            letters = ['A', 'B', 'C', 'D', 'E', 'F']
        for idcs_ax, letter in zip(idcs_axs, letters):
            self.ax[idcs_ax].text(-0.1, 1, letter, fontweight='bold', transform=self.ax[idcs_ax].transAxes, fontsize=14)

        if not sub_figure:
            ax_offset = self.ax.shape[0] - 2
            bbox = dict(facecolor=transparent_to_opaque('#f7eacf', 0.4), edgecolor='black', boxstyle='round')
            x_offset = 0.25
            self.ax[0 + ax_offset, 0].text(x_offset, 0.94, 'difficulty\neffect', transform=self.ax[0 + ax_offset, 0].transAxes, fontsize=10, va='center',
                          ha='center', bbox=bbox)
            self.ax[1 + ax_offset, 0].text(x_offset, 0.94, 'difficulty\neffect', transform=self.ax[1 + ax_offset, 0].transAxes, fontsize=10, va='center',
                          ha='center', bbox=bbox)
            self.ax[0 + ax_offset, 1].text(x_offset + 0.1, 0.94, 'interaction difficulty-\nfeedback effect', transform=self.ax[0 + ax_offset, 1].transAxes, fontsize=10, va='center', ha='center',
                          bbox=bbox)
            self.ax[1 + ax_offset, 1].text(x_offset + 0.1, 0.94, 'interaction difficulty-\nfeedback effect', transform=self.ax[1 + ax_offset, 1].transAxes, fontsize=10, va='center', ha='center',
                          bbox=bbox)

    def show_plot(self, savefig_name=None, pad=0, tight_layout=True, bbox_tight=True):
        if savefig_name and self.dir_figures is not None:
            os.makedirs(self.dir_figures, exist_ok=True)
            plt.savefig(os.path.join(self.dir_figures, f'{savefig_name}.pdf'),
                        bbox_inches=[None, 'tight'][bbox_tight], pad_inches=[None, 0.0][bbox_tight], format='pdf', dpi=300)
        if tight_layout:
            plt.tight_layout(pad=pad)
        plt.show()

    @staticmethod
    def add_legend(colors, linestyles, labels, plot_axis, loc=None):
        """Add customized legend to plot."""
        xlims = plot_axis.get_xlim()
        ylims = plot_axis.get_ylim()
        custom_lines = []
        for color, linestyle in zip(colors, linestyles):
            if linestyle == 'full':
                custom_lines.append(Patch(facecolor=color))
            elif linestyle == 'x':
                custom_lines.append(Line2D([0], [0], color=color, linestyle='-', linewidth=0))
            elif (linestyle == 'o') or (linestyle == 'v') or (linestyle == '.') or (linestyle == '*') or (linestyle == 'D'):
                custom_lines.append(plot_axis.scatter(1e9, 0, marker=linestyle, s=25, color=color))
            else:
                custom_lines.append(Line2D([0], [0], color=color, linestyle=linestyle))
        plot_axis.legend(custom_lines, labels, loc=loc)
        plot_axis.set_xlim(xlims)
        plot_axis.set_ylim(ylims)

    @staticmethod
    def power_law(x, A, b):
        return A * (x ** b)

    @staticmethod
    def model_effect_size_breakdown_log(x, C, m, a):
        # return np.log(C + C * a * x ** m)
        return np.log(C + (x * (C ** (1 / m) / a)) ** m)
    
    @staticmethod
    def model_effect_size_breakdown(x, C, m, a):
        # return C + C * a * x ** m
        return C + (x * (C ** (1 / m) / a)) ** m


# simulate effect sizes
sim = EffectSizeSimulator(dir_save=os.path.join('results', 'effect_size_analysis'))
effect_sizes = np.geomspace(1e-4, 0.5, 20)  # proportion of RT variance
nums_participants = np.round(np.geomspace(5, 100, 10)).astype(int)
f2_stim_acc_ded, f2_stim_acc_ded_interaction_neg_acc_ind = sim.get_f2(effect_sizes, nums_participants)

# plot simulation results
plotting = PlotResults(f2_stim_acc_ded, f2_stim_acc_ded_interaction_neg_acc_ind, nums_participants,
                       dir_figures='figures')
plotting.fit_effect_size_breakdown()
plotting.fit_min_detectable_effect_sizes()
plotting.plot_all()
