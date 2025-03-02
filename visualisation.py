import os
import numpy as np
from scipy.stats import sem
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.axisartist.axislines import SubplotZero

from utils import add_legend, transparent_to_opaque


class PlotAllResults(object):
    """
    Class to plot post trial effects and model results in comparison to experimental data.

    Parameters
    ----------
    df_model_basic : pd.DataFrame
        Dataframe containing the basic model results
    df_model_extended : pd.DataFrame
        Dataframe containing the extended model results
    df_paired : pd.DataFrame
        Dataframe containing the paired data
    dir_figures : str
        Directory to save the figures
    savefigs : bool
        Whether to save the figures

    Attributes
    ----------
    df_model_basic : pd.DataFrame
        Dataframe containing the basic model results
    df_model_extended : pd.DataFrame
        Dataframe containing the extended model results
    df_paired : pd.DataFrame
        Dataframe containing the paired data
    num_participants : int
        Number of participants
    dir_figures : str
        Directory to save the figures
    savefigs : bool
        Whether to save the figures
    fig : plt.Figure
        Figure object
    ax : np.ndarray
        Array of axes objects
    colors : list
        List of colors
    acc_conds : list
        List of accuracy conditions
    x_offset_plot : float
        Offset for plotting the individual means
    errorbar_handle : np.ndarray
        Array of errorbar handles
    """
    def __init__(self, df_model_basic=None, df_model_extended=None, df_paired=None, dir_figures='figures', savefigs=True):
        for df in [df_model_basic, df_model_extended]:
            df.params *= 1000
            df.bse *= 1000
        self.df_model_basic = df_model_basic
        self.df_model_extended = df_model_extended
        self.df_paired = df_paired
        self.num_participants = len(df_paired['part'].unique())
        self.dir_figures = dir_figures
        self.savefigs = savefigs

        os.makedirs(self.dir_figures, exist_ok=True)

        self.fig, self.ax = plt.subplots(figsize=(8.4, 8.4), nrows=2, ncols=2, width_ratios=[1, 1],
                                    height_ratios=[1, 1])
        self.colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        self.acc_conds = [0.35, 0.75]
        self.x_offset_plot = 0.15
        self.errorbar_handle = np.zeros(2, dtype=object)

    def show(self):
        for idx_acc_cond, acc_cond in enumerate(self.acc_conds):
            self.show_post_error_slowing(idx_acc_cond, acc_cond)

        for idx_target_acc, target_acc in enumerate([0.35, 0.75]):
            self.show_RT_curve_condition(idx_target_acc, target_acc)

        self.add_axis_labels()
        self.set_ticks()
        self.add_inset_titles()
        self.set_frame_details()
        self.add_legend()
        self.show_plot(subfigures=True, legend=False, legend_loc=3,
                       savefig_name='RT_curve_paper',
                       tight_layout=False, bbox_tight=True)

    def show_post_error_slowing(self, idx_acc_cond, acc_cond, plot_individual_means=True):
        RTs_post_error, RTs_post_correct, num_post_error, num_post_correct = (
            self.compute_RTs_post_trial(acc_cond))

        if plot_individual_means:
            self.plot_individual_means(idx_acc_cond, RTs_post_correct, RTs_post_error)

        self.plot_errorbars(idx_acc_cond, acc_cond)

        self.add_labeled_arrows_PES(idx_acc_cond, RTs_post_correct, RTs_post_error)

        self.ax[0, idx_acc_cond].set_xlim([0, 1])
        self.ax[0, idx_acc_cond].set_ylim([435, 670])

    def compute_RTs_post_trial(self, acc_cond):
        RTs_post_error = np.zeros(self.num_participants)
        RTs_post_correct = np.zeros(self.num_participants)
        num_post_error = np.zeros(self.num_participants)
        num_post_correct = np.zeros(self.num_participants)

        for idx_part in range(self.num_participants):
            mask_post_error = ((self.df_paired['part'] == idx_part)
                           & (self.df_paired['acc_ded'] == 1)
                           & (self.df_paired['target_acc'] == acc_cond)
                           & (self.df_paired['acc_ind'] == 0)
                           & (self.df_paired['col_ded'] >= 0))
            mask_post_correct = ((self.df_paired['part'] == idx_part)
                           & (self.df_paired['acc_ded'] == 1)
                           & (self.df_paired['target_acc'] == acc_cond)
                           & (self.df_paired['acc_ind'] == 1)
                           & (self.df_paired['col_ded'] >= 0))

            y_RT_data_error = self.df_paired['RT_ded'][mask_post_error]
            for idx_color_ded, factor in zip([1, 2, 3], ['green', 'blue', 'yellow']):
                y_RT_data_error[self.df_paired['col_ded'][mask_post_error] == idx_color_ded] \
                    -= self.df_model_basic.params[factor] / 1000

            y_RT_data_correct = self.df_paired['RT_ded'][mask_post_correct]
            for idx_color_ded, factor in zip([1, 2, 3], ['green', 'blue', 'yellow']):
                y_RT_data_correct[self.df_paired['col_ded'][mask_post_correct] == idx_color_ded] \
                    -= self.df_model_basic.params[factor] / 1000

            RTs_post_error[idx_part] = np.mean(y_RT_data_error)
            RTs_post_correct[idx_part] = np.mean(y_RT_data_correct)
            num_post_error[idx_part] = len(y_RT_data_error)
            num_post_correct[idx_part] = len(y_RT_data_correct)

        return RTs_post_error, RTs_post_correct, num_post_error, num_post_correct

    def plot_individual_means(self, idx_acc_cond, RTs_post_correct, RTs_post_error):
        self.ax[0, idx_acc_cond].scatter(np.full(RTs_post_correct.shape, fill_value=0.5 + self.x_offset_plot),
                                         RTs_post_correct * 1000,
                                         color=plt.rcParams['axes.prop_cycle'].by_key()['color'][0],
                                         alpha=0.5)
        self.ax[0, idx_acc_cond].scatter(np.full(RTs_post_error.shape, fill_value=0.5 - self.x_offset_plot),
                                         RTs_post_error * 1000,
                                         color=plt.rcParams['axes.prop_cycle'].by_key()['color'][1],
                                         alpha=0.5)

    def plot_errorbars(self, idx_acc_cond, acc_cond):
        if acc_cond == 0.35:
            y_post_error = self.df_model_basic.params['Intercept']
            y_post_correct = (self.df_model_basic.params['Intercept']
                              + self.df_model_basic.params['acc_ind:target_acc_35'])
            y_post_error_SE = self.df_model_basic.bse['Intercept']
            y_post_correct_SE = np.sqrt(self.df_model_basic.bse['Intercept']**2
                                        + self.df_model_basic.bse['acc_ind:target_acc_35']**2)
        elif acc_cond == 0.75:
            y_post_error = (self.df_model_basic.params['Intercept']
                            + self.df_model_basic.params['neg_acc_ind:target_acc_75']
                            + self.df_model_basic.params['target_acc_75'])
            y_post_correct = self.df_model_basic.params['Intercept'] + self.df_model_basic.params['target_acc_75']
            y_post_error_SE = np.sqrt(self.df_model_basic.bse['Intercept']**2
                                      + self.df_model_basic.bse['neg_acc_ind:target_acc_75']**2
                                      + self.df_model_basic.bse['target_acc_75']**2)
            y_post_correct_SE = np.sqrt(self.df_model_basic.bse['Intercept']**2
                                        + self.df_model_basic.bse['target_acc_75']**2)

        self.ax[0, idx_acc_cond].errorbar(0.5 + self.x_offset_plot, y_post_correct, yerr=y_post_correct_SE,
                                          capsize=10, color='black', fmt='s', markersize=4)
        self.ax[0, idx_acc_cond].errorbar(0.5 - self.x_offset_plot, y_post_error, yerr=y_post_error_SE,
                                          capsize=10, color='black', fmt='s', markersize=4)

    def add_labeled_arrows_PES(self, idx_acc_cond, RTs_post_correct, RTs_post_error):
        arrowprops = dict(arrowstyle="->, head_width=0.4, head_length=0.6",
                          connectionstyle="arc3",
                          lw=1.5)
        arrowprops['color'] = plt.rcParams['axes.prop_cycle'].by_key()['color'][idx_acc_cond]
        if idx_acc_cond == 0:
            x1 = 0.5 + self.x_offset_plot
            x2 = 0.5 - self.x_offset_plot
            y1 = np.mean(RTs_post_correct) * 1000
            y2 = np.mean(RTs_post_error) * 1000
        else:
            x1 = 0.5 - self.x_offset_plot
            x2 = 0.5 + self.x_offset_plot
            y1 = np.mean(RTs_post_error) * 1000
            y2 = np.mean(RTs_post_correct) * 1000
        offset_percent = 0.1
        self.ax[0, idx_acc_cond].annotate("", xy=(x1 + (x2 - x1) * offset_percent, y1 + (y2 - y1) * offset_percent), xycoords='data',
                        xytext=(x1 + (x2 - x1) * (1 - offset_percent), y1 + (y2 - y1) * (1 - offset_percent)), textcoords='data',
                        arrowprops=arrowprops)
        if idx_acc_cond == 0:
            self.ax[0, idx_acc_cond].text(0.5, y1 + (y2 - y1) * 0.5 - 15, 'PCS', color='black', va='center', ha='center')
        else:
            self.ax[0, idx_acc_cond].text(0.5, y1 + (y2 - y1) * 0.5 - 15, 'PES', color='black', va='center', ha='center')

    def show_RT_curve_condition(self, idx_target_acc, target_acc):
        y_x_min = []
        y_x_max = []
        for idx_condition, condition in enumerate(['post-correct', 'post-error']):
            x_acc_bins_selected, y_RT_bins_selected, y_RT_bins_sem_selected = self.get_RT_bins(target_acc, condition)

            self.errorbar_handle[idx_condition] = self.plot_errorbars_curve(
                idx_target_acc, idx_condition, x_acc_bins_selected, y_RT_bins_selected, y_RT_bins_sem_selected)

            x_acc = np.linspace(min(x_acc_bins_selected), max(x_acc_bins_selected), 1000)
            y_RT_model = self.get_y_RT_model(x_acc, target_acc, condition)
            self.ax[1, idx_target_acc].plot(x_acc * 100, y_RT_model, color=self.colors[idx_condition], linestyle='solid',
                                    zorder=[5, 0][idx_condition])

            y_x_min.append(y_RT_model[0])
            y_x_max.append(y_RT_model[-1])

        self.ax[1, idx_target_acc].set_xlim([25, 100])
        self.ax[1, idx_target_acc].set_ylim([465, 630])

        self.add_labeled_arrows_curves(idx_target_acc, x_acc, y_x_min, y_x_max)

    def get_RT_bins(self, target_acc, condition):
        mask_select = ((self.df_paired['acc_ded'] == 1)
                       & (self.df_paired['target_acc'] == target_acc)
                       & (self.df_paired['acc_ind'] == int(condition == 'post-correct'))
                       & (self.df_paired['col_ded'] >= 0))

        x_acc_data = self.df_paired['stim_acc_ded'][mask_select]  # self.stim_accs[mask_select]

        y_RT_data = self.df_paired['RT_ded'][mask_select]
        for idx_color_2, factor in zip([1, 2, 3], ['green', 'blue', 'yellow']):
            y_RT_data[self.df_paired['col_ded'][mask_select] == idx_color_2] -= self.df_model_extended.params[
                                                                                    factor] / 1000

        x_acc_bins = np.arange(0.35, 1.0, 0.05)
        y_RT_bins = np.zeros(x_acc_bins.shape)
        y_RT_bins_num_data_points = np.zeros(x_acc_bins.shape)
        y_RT_bins_sem = np.zeros(x_acc_bins.shape)
        for idx_bin, (x_acc_low, x_acc_high) in enumerate(zip(x_acc_bins[:-1], x_acc_bins[1:])):
            y_RT_bins[idx_bin] = np.mean(y_RT_data[(x_acc_data >= x_acc_low)
                                                   & (x_acc_data < x_acc_high)])
            y_RT_bins_num_data_points[idx_bin] = sum((x_acc_data >= x_acc_low) & (x_acc_data < x_acc_high))
            y_RT_bins_sem[idx_bin] = sem(y_RT_data[(x_acc_data >= x_acc_low)
                                                   & (x_acc_data < x_acc_high)])

        # remove bins with less than 15 data points
        min_samplesize = 15
        x_acc_bins_selected = x_acc_bins[y_RT_bins_num_data_points >= min_samplesize]
        y_RT_bins_selected = y_RT_bins[y_RT_bins_num_data_points >= min_samplesize]
        y_RT_bins_sem_selected = y_RT_bins_sem[y_RT_bins_num_data_points >= min_samplesize]

        return x_acc_bins_selected, y_RT_bins_selected, y_RT_bins_sem_selected

    def plot_errorbars_curve(self, idx_target_acc, idx_condition,
                             x_acc_bins_selected, y_RT_bins_selected, y_RT_bins_sem_selected):
        return self.ax[1, idx_target_acc].errorbar(
            x_acc_bins_selected * 100,
            y_RT_bins_selected * 1000,
            yerr=y_RT_bins_sem_selected * 1000,
            capsize=10, color=transparent_to_opaque(self.colors[idx_condition], alpha=0.5), fmt='s',
            label=['post-correct', 'post-error'][idx_condition], zorder=[-5, -10][idx_condition])

    def get_y_RT_model(self, x_acc, target_acc, condition):
        y_RT_model = self.df_model_extended.params['Intercept'] * np.ones(x_acc.shape)
        if target_acc == 0.75:
            y_RT_model += self.df_model_extended.params['target_acc_75'] * np.ones(x_acc.shape)
        if condition == 'post-error':
            y_RT_model += self.df_model_extended.params['neg_acc_ind'] * np.ones(x_acc.shape)
            y_RT_model += self.df_model_extended.params['neg_acc_ind:stim_acc_ded'] * x_acc
            if target_acc == 0.75:
                y_RT_model += self.df_model_extended.params['neg_acc_ind:target_acc_75'] * np.ones(x_acc.shape)
        y_RT_model += self.df_model_extended.params['stim_acc_ded'] * x_acc
        y_RT_model += self.df_model_extended.params['stim_acc_ded_sq_scaled'] * x_acc ** 2 * 0.01
        y_RT_model += self.df_model_extended.params['stim_acc_ded_cu_scaled'] * x_acc ** 3 * 0.0001

        return y_RT_model

    def add_labeled_arrows_curves(self, idx_target_acc, x_acc, y_x_min, y_x_max):
        arrowprops = dict(arrowstyle="->, head_width=0.4, head_length=0.6",
                          connectionstyle="arc3",
                          lw=1.5)
        arrowprops['color'] = self.colors[0]
        self.ax[1, idx_target_acc].annotate("",
                    xy=([32.5, 32.5][idx_target_acc], y_x_min[0] + 1.5), xycoords='data',
                    xytext=([32.5, 32.5][idx_target_acc], y_x_min[1] - 1.5), textcoords='data',
                    arrowprops=arrowprops)
        arrowprops['color'] = self.colors[1]
        self.ax[1, idx_target_acc].annotate("",
                    xy=([82.5, 92.5][idx_target_acc], y_x_max[1] + 1.5), xycoords='data',
                    xytext=([82.5, 92.5][idx_target_acc], y_x_max[0] - 1.5), textcoords='data',
                    arrowprops=arrowprops)

        self.ax[1, idx_target_acc].text([max(x_acc * 100) + 6, max(x_acc * 100) + 6][idx_target_acc],
                                [np.mean(y_x_max), np.mean(y_x_max)][idx_target_acc], 'PES',
                                color='black', rotation=90, va='center', ha='center')
        self.ax[1, idx_target_acc].text([min(x_acc * 100) - 6, min(x_acc * 100) - 6][idx_target_acc],
                                [np.mean(y_x_min), np.mean(y_x_min)][idx_target_acc],
                                'PCS', color='black', rotation=90, va='center', ha='center')

    def add_axis_labels(self):
        self.ax[1, 0].set_xlabel('stimulus difficulty [% correct]')
        self.ax[1, 1].set_xlabel('stimulus difficulty [% correct]')
        self.ax[0, 0].set_ylabel('reaction time (red-centered) [ms]')
        self.ax[1, 0].set_ylabel('reaction time (red-centered) [ms]')

    def set_ticks(self):
        for idx_acc_cond in range(2):
            self.ax[0, idx_acc_cond].set_xticks([0.5 - self.x_offset_plot, 0.5 + self.x_offset_plot])
            self.ax[0, idx_acc_cond].set_xticklabels(['post-error', 'post-correct'])
        self.ax[0, 1].set_yticklabels([])
        self.ax[1, 1].set_yticklabels([])

    def add_inset_titles(self):
        for idcs_ax, letter in zip([(0, 0), (0, 1), (1, 0), (1, 1)], ['A', 'B', 'C', 'D']):
            self.ax[idcs_ax].text(-0.1, 1, letter, fontweight='bold', transform=self.ax[idcs_ax].transAxes, fontsize=14)

        bbox = dict(facecolor=transparent_to_opaque('#f7eacf', 0.4), edgecolor='black', boxstyle='round')
        self.ax[0, 0].text(0.15, 0.94, 'difficult\ncontext', transform=self.ax[0, 0].transAxes, fontsize=10, va='center',
                      ha='center', bbox=bbox)
        self.ax[1, 0].text(0.15, 0.94, 'difficult\ncontext', transform=self.ax[1, 0].transAxes, fontsize=10, va='center',
                      ha='center', bbox=bbox)
        self.ax[0, 1].text(0.15, 0.94, 'easy\ncontext', transform=self.ax[0, 1].transAxes, fontsize=10, va='center', ha='center',
                      bbox=bbox)
        self.ax[1, 1].text(0.15, 0.94, 'easy\ncontext', transform=self.ax[1, 1].transAxes, fontsize=10, va='center', ha='center',
                      bbox=bbox)

    def set_frame_details(self):
        self.fig.subplots_adjust(top=0.9, wspace=0.05)

        for ax in self.ax.flatten():
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)

    def add_legend(self):
        legend_elements = [
            self.errorbar_handle[0],
            self.errorbar_handle[1],
            Line2D([0], [0], color=self.colors[0], ls='-', lw=2, label='post-corr. fit'),
            Line2D([0], [0], color=self.colors[1], ls='-', lw=2, label='post-error fit')
        ]
        self.ax[1, 0].legend(handles=legend_elements, loc=['lower left', 'lower left'][0])

    def show_plot(self, title=None, subfigures=False, legend=False, legend_loc='', savefig_name=None, pad=1.08,
                  tight_layout=True, bbox_tight=False):
        if title:
            if subfigures:
                self.fig.suptitle(title)
            else:
                plt.title(title)
        if legend:
            if legend_loc:
                plt.legend(loc=legend_loc)
            else:
                plt.legend()
        if savefig_name and self.savefigs:
            plt.savefig(os.path.join(self.dir_figures, 'RT_curve_PCS_study.pdf'),
                        bbox_inches=[None, 'tight'][bbox_tight], format='pdf', dpi=300)
        if tight_layout:
            plt.tight_layout(pad=pad)
        plt.show()


class PlotTheory(object):
    """
    Class to visualise the theoretical hypothesis explaining the results.

    Parameters
    ----------
    dir_figures : str
        Directory to save the figures

    Attributes
    ----------
    dir_figures : str
        Directory to save the figures
    fig : plt.Figure
        Figure object
    ax : SubplotZero
        Axis object
    colors : list
        List of colors
    num : int
        Number of points
    x : np.ndarray
        Array of x values
    origin_offset_x : int
        Offset for the x-axis
    y_offset : float
        Offset for the y-axis
    ylims : list
        List of y limits
    """
    def __init__(self, dir_figures='figures'):
        self.dir_figures = dir_figures

        self.fig = plt.figure(figsize=(5.4, 5.4 / 6.4 * 4.8))
        self.ax = SubplotZero(self.fig, 111)
        self.fig.add_subplot(self.ax)

        self.colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

        self.num = 1000
        self.x = np.linspace(0, 100, self.num)
        self.origin_offset_x = 10
        self.y_offset = 0.03
        self.ylims = [0, 0.525]

    def show(self):
        self.plotting()
        self.set_ticks()
        self.set_axis_style()
        self.set_text()
        self.save_plot()

    def plot_curves(self):
        half = self.num // 2
        self.ax.plot(self.x[:half] + self.origin_offset_x, self.inv_u(self.x)[:half], color=self.colors[1])
        self.ax.plot(self.x[:half] + self.origin_offset_x, self.inv_u(self.perception_change(self.x))[:half], color=self.colors[0], linestyle='dashed')
        self.ax.plot(self.x[half:] + self.origin_offset_x, self.inv_u(self.x)[half:], color=self.colors[0])
        self.ax.plot(self.x[half:] + self.origin_offset_x, self.inv_u(self.perception_change(self.x))[half:], color=self.colors[1], linestyle='dashed')

    def set_ticks(self):
        self.ax.set_xticks([12.5 + self.origin_offset_x, 50 + self.origin_offset_x, 87.5 + self.origin_offset_x], minor=False) # 7.5, 92.5
        self.ax.set_xticklabels(['\ndifficult', '\nintermediate', '\neasy'])

        self.ax.set_yticks([])

    def set_axis_style(self):
        self.ax.axis['xzero'].set_axisline_style("-|>")
        self.ax.axis['xzero'].set_visible(True)
        self.ax.axis['yzero'].set_axisline_style("-|>")
        self.ax.axis['yzero'].set_visible(True)
        for direction in ["left", "right", "bottom", "top"]:
            self.ax.axis[direction].set_visible(False)

        self.ax.set_xlim([-10 + self.origin_offset_x, 110 + self.origin_offset_x])
        self.ax.set_ylim(self.ylims)

    def set_text(self):
        self.ax.text(60, -0.07, 'trial stimulus difficulty', ha='center')
        self.ax.set_ylabel('reaction time')
        add_legend([self.colors[0], self.colors[1], 'dimgray', 'dimgray'],
                   ['full', 'full', 'solid', 'dashed'],
                   ['post-correct', 'post-error', 'typical', 'atypical'], self.ax, loc='center')

    def plotting(self):
        self.plot_curves()
        for mode, x_orig, x_slowing_arrow, idx_col in zip(['PCS', 'PES'], [12.5, 100 - 12.5], [12.5, 100 - 12.5], [0, 1]):
            props = dict(boxstyle='round', facecolor='white', edgecolor=self.colors[idx_col], alpha=1)

            # pathway no change in perception
            self.ax.scatter(x_orig + self.origin_offset_x, self.ylims[0], color='dimgray', clip_on=False, zorder=100)
            self.ax.vlines(x_orig+ self.origin_offset_x, self.ylims[0], self.inv_u(x_orig), color='dimgray', linestyle='dotted', linewidth=1)
            self.ax.scatter(x_orig+ self.origin_offset_x, self.inv_u(x_orig), color=self.colors[(idx_col + 1) % 2], zorder=100)

            # pathway change in perception
            self.ax.annotate("", xy=(self.perception_change(x_orig)+ self.origin_offset_x, self.ylims[0]), xytext=(x_orig+ self.origin_offset_x, self.ylims[0]),
                        arrowprops=dict(ec=self.colors[idx_col], fc=self.colors[idx_col], headwidth=5, headlength=5, width=1,
                                   shrink=0.175))
            self.ax.scatter(self.perception_change(x_orig) + self.origin_offset_x, self.ylims[0], color=self.colors[idx_col], alpha=0.4, clip_on=False, zorder=100)
            self.ax.text(x_orig + (self.perception_change(x_orig) - x_orig) + self.origin_offset_x, self.ylims[0]+self.y_offset, 'perceived\ndifficulty', ha='center', bbox=props)
            self.ax.vlines(self.perception_change(x_orig) + self.origin_offset_x, self.ylims[0], self.inv_u(self.perception_change(x_orig)), color='dimgray', linestyle='dotted', linewidth=1)
            self.ax.scatter(self.perception_change(x_orig) + self.origin_offset_x, self.inv_u(self.perception_change(x_orig)), color=self.colors[idx_col], alpha=0.4, zorder=100)
            self.ax.annotate("", xy=(x_orig + self.origin_offset_x, self.inv_u(self.perception_change(x_orig))), xytext=(self.perception_change(x_orig) + self.origin_offset_x, self.inv_u(self.perception_change(x_orig))),
                        arrowprops=dict(ec='dimgray', fc='dimgray', headwidth=5, headlength=5, width=1, shrink=0.175))
            self.ax.scatter(x_orig + self.origin_offset_x, self.inv_u(self.perception_change(x_orig)), color=self.colors[idx_col], zorder=100)
            self.ax.text(x_orig + self.origin_offset_x, self.inv_u(self.perception_change(x_orig)) + self.y_offset, 'actual\ndifficulty', ha='center', bbox=props)

            # slowing arrows
            self.ax.annotate("", xy=(x_slowing_arrow + self.origin_offset_x, self.inv_u(self.perception_change(x_slowing_arrow))), xytext=(x_slowing_arrow + self.origin_offset_x, self.inv_u(x_slowing_arrow)),
                        arrowprops=dict(ec=self.colors[idx_col], fc=self.colors[idx_col], headwidth=5, headlength=5, width=1,
                              shrink=0.2))
            self.ax.text(x_slowing_arrow + [-2.9, 3.5][int(mode == 'PES')] + self.origin_offset_x, (self.inv_u(x_slowing_arrow) + self.inv_u(self.perception_change(x_slowing_arrow))) / 2 - 0.0025,
                    mode, ha='center', va='center', rotation=90)

    def save_plot(self):
        plt.savefig(os.path.join(self.dir_figures, 'explanation_slowing_effects.pdf'))
        plt.show()

    @staticmethod
    def inv_u(x):
        return -((x - 50) / 100) ** 2 + 0.45

    @staticmethod
    def perception_change(x, scale=1.5):
        return (x - 50) / scale + 50
