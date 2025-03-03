import subprocess
import csv
import numpy as np
import re
import os
import pandas as pd
from scipy.stats import iqr

from utils import get_files, norm_cdf_mAFC, normalize_data


class DataProcessing(object):
    """
    Class to reshape and process the data from the post-correct slowing study.

    Parameters
    ----------
    dir_data : str
        Directory containing the raw data files
    dir_data_prep : str
        Directory to save the preprocessed data files
    dir_data_psychometric_curves : str
        Directory containing the psychometric curve data files
    verbose : bool
        Whether to print information during the data processing

    Attributes
    ----------
    dir_data : str
        Directory containing the raw data files
    dir_data_prep : str
        Directory to save the preprocessed data files
    dir_data_psychometric_curves : str
        Directory containing the psychometric curve data files
    verbose : bool
        Whether to print information during the data processing
    data_files : dict
        Dictionary containing the data files
    participants : np.array
        Array containing the participant IDs
    num_parts : int
        Number of participants
    key_to_idx : dict
        Dictionary mapping key responses to indices
    col_to_gen_idx : dict
        Dictionary mapping colors to generic indices
    num_colors : int
        Number of colors
    masks : dict
        Dictionary containing masks for data selection
    data_prep : dict
        Dictionary containing the preprocessed data
    part_outliers : list
        List of outlier participants
    df_all : pd.DataFrame
        Dataframe containing the preprocessed data
    df_paired : pd.DataFrame
        Dataframe containing the paired data
    data : dict
        Dictionary containing the data
    """

    def __init__(self, dir_data='data/data_raw', dir_data_prep='data/data_preprocessed',
                 dir_data_psychometric_curves='data/glmer_psychometric_curves', verbose=True):
        self.dir_data = dir_data
        self.dir_data_prep = dir_data_prep
        self.dir_data_psychometric_curves = dir_data_psychometric_curves
        self.verbose = verbose

        os.makedirs(self.dir_data_prep, exist_ok=True)
        os.makedirs(self.dir_data_psychometric_curves, exist_ok=True)

        file_list = get_files(dir_data)
        self.data_files = self.get_participants(file_list, dir_data)
        self.participants = np.array(sorted(self.data_files.keys()))
        self.num_parts = len(self.participants)

        # Dictionaries with which to map key responses or indices to colors and generic indices.
        # Use generic indices (gen_idx) to make, e.g., "blue selected" have the same index for all participants during
        # the analysis
        self.key_to_idx = {'d': 0, 'f': 1, 'j': 2, 'k': 3}
        self.col_to_gen_idx = {'r': 0, 'g': 1, 'b': 2, 'y': 3}  # gen_idx = generic index
        self.num_colors = len(self.col_to_gen_idx)

        self.masks = {}
        self.data_prep = {}
        self.part_outliers = []

        self.df_all = None
        self.df_paired = None

        self.data = self.read_data()

    def read_data(self):
        data = {}
        trial_id_row = 0
        self.columns = []
        for participant in self.participants:
            data_participant = self.read_data_part(participant, trial_id_row)
            self.columns.extend(data_participant.keys())

            for key in data_participant.keys():
                try:
                    data[(key, participant)] = np.array(data_participant[key])
                except ValueError:
                    pass
        return data

    def read_data_part(self, participant, trial_id_row):
        data_participant = {'trial_id': []}
        trial_id = 1

        with open(self.data_files[participant], 'rt', encoding='utf-8-sig') as csvfile:
            csv_reader = csv.reader(csvfile, delimiter=',')
            header = next(csv_reader)
            for idx_column, key in enumerate(header):
                if key not in data_participant.keys():
                    data_participant[key] = []

            for idx_row, row in enumerate(csv_reader):
                if not row[trial_id_row]:  # row trial_id_row is trial_id
                    continue  # skip the row that signals end of experiment

                for idx_key, key in enumerate(header):
                    if not row[idx_key]:
                        data_participant[key].append(None)  # fill empty values with None
                        continue

                    try:
                        number = float(row[idx_key])
                        if number.is_integer():
                            number = int(number)
                        data_participant[key].append(number)
                    except ValueError:
                        data_participant[key].append(row[idx_key])

                if 'trial_id' not in header:
                    data_participant['trial_id'].append(trial_id)
                    trial_id += 1
        return data_participant

    def get_num_trials(self):
        # Calculate first the number of ind / ded training / testing trials
        mask_mode_train = self.data[('mode', self.participants[0])] == 'training'
        mask_mode_test = self.data[('mode', self.participants[0])] == 'testing'
        num_trials = {
            'all_train': sum(entry is not None
                             for entry in self.data[('color_idx_ind', self.participants[0])][mask_mode_train]),
            'all_test': sum(entry is not None
                            for entry in self.data[('color_idx_ind', self.participants[0])][mask_mode_test]),
            'all': sum(entry is not None
                            for entry in self.data[('color_idx_ind', self.participants[0])])
        }
        return num_trials

    def prepare_data(self, data_mode):
        # This method restructures the data from the raw data files into a more useful trial-based format for the
        # analysis, removing unsuitable trials.

        self.num_trials = self.get_num_trials()

        # Start to reshape data into a more useful format.
        # Note: one inductive and one deductive trial together count as two trials from here on

        # Create useful masks with which to select data subsets
        self.masks = {
            'all_testing': np.full((self.num_parts, self.num_trials['all_test']), False),
            'core_testing': np.full((self.num_parts, self.num_trials['all_test']), False),
            'outliers_RT': np.full((self.num_parts, self.num_trials['all_test']), False),
            'new_block': np.full((self.num_parts, self.num_trials['all_test']), False),
            'timely_response': np.full((self.num_parts, self.num_trials['all_test']), True)
        }

        # Store useful experiment data in arrays
        self.data_prep = {
            'colors_idx': np.full((self.num_parts, self.num_trials['all_test']), np.nan, dtype=int),
            'target_accs': np.full((self.num_parts, self.num_trials['all_test']), np.nan),
            'select_colors_idx': np.full((self.num_parts, self.num_trials['all_test']), np.nan),
            'trials_corr': np.full((self.num_parts, self.num_trials['all_test']), np.nan, dtype=bool),
            'RTs': np.full((self.num_parts, self.num_trials['all_test']), np.nan),
            'RTs_norm': np.full((self.num_parts, self.num_trials['all_test']), np.nan),
            'stim_colors_rgb': np.full((self.num_parts, self.num_trials['all_test'], 3), np.nan),
            'stim_colors_max': np.full((self.num_parts, self.num_trials['all_test']), np.nan),
            'stim_accs': np.full((self.num_parts, self.num_trials['all_test']), np.nan)
        }

        # loop over all participants and fill in the data arrays
        for idx_part, part in enumerate(self.participants):

            # Note: deductive trial doesn't count as an individual trial!
            self.num_all_full_trials = len(self.data[('trial_id', part)])
            if self.verbose:
                print('Processing participant:', part)
                print('# all trials:', self.num_all_full_trials)
                print('')

            self.fill_data_arrays_part(idx_part, part, data_mode=data_mode)

    def fill_data_arrays_part(self, idx_part, part, data_mode='testing'):
        idx_to_col = {i: self.data[('color_mapping', part)][0][i] for i in range(self.num_colors)}

        # Loop over all trials and fill the data arrays which will be used during the analysis.
        # To simplify things, fill in inductive and deductive trials as separate trials
        # (as is not the case in the actual data files)
        test_trial_idx = 0
        for idx_trial in range(self.num_all_full_trials):
            if not (self.data[('mode', part)][idx_trial] == data_mode):
                continue
            self.masks['all_testing'][idx_part, test_trial_idx] = int(self.data[('mode', part)][idx_trial] == 'testing')

            # Get the relevant data for each trial
            target_acc = self.data[('target_acc', part)][idx_trial]
            color_idx_ind = self.data[('color_idx_ind', part)][idx_trial]
            key_resp_ind_key = self.data[('key_resp_ind.keys', part)][idx_trial]
            key_resp_ind_rt = self.data[('key_resp_ind.rt', part)][idx_trial]
            stim_ind_color_rgb = np.array((self.data[('stim_ind_color_rgb_r', part)][idx_trial],
                                           self.data[('stim_ind_color_rgb_g', part)][idx_trial],
                                           self.data[('stim_ind_color_rgb_b', part)][idx_trial]))
            trials_thisN = self.data[('trials.thisN', part)][idx_trial]

            # Determine block change by comparing variables that change after a block change to the previous trial
            if idx_trial > 0:
                trial_after_break = ((self.data[('text_break_block.started', part)][idx_trial - 1] is not None)
                                     & (self.data[('text_break_block.started', part)][idx_trial - 1] != 'None'))
                trial_block_change = ((self.data[('trials_staircase.thisN', part)][idx_trial] is not None)
                                          != (self.data[('trials_staircase.thisN', part)][idx_trial - 1] is not None)) \
                                         | ((self.data[('trials.thisN', part)][idx_trial] is not None)
                                            != (self.data[('trials.thisN', part)][idx_trial - 1] is not None))

                trial_mode_change = (self.data[('mode', part)][idx_trial]
                                      != self.data[('mode', part)][idx_trial - 1])
                trial_resp_deadline_change = (self.data[('response_deadline', part)][idx_trial]
                                      != self.data[('response_deadline', part)][idx_trial - 1])
            else:
                trial_after_break = False
                trial_block_change = False
                trial_mode_change = False
                trial_resp_deadline_change = False

            # Fill in the values both for the inductive and the deductive trials trial by trial
            if color_idx_ind is not None:
                self.data_prep['target_accs'][idx_part, test_trial_idx] = target_acc
                if trials_thisN is not None:
                    self.masks['core_testing'][idx_part, test_trial_idx] = True  # test trials are numbered by thisN
                self.data_prep['colors_idx'][idx_part, test_trial_idx] = self.col_to_gen_idx[idx_to_col[color_idx_ind]]
                if key_resp_ind_key != 'None':
                    self.data_prep['select_colors_idx'][idx_part, test_trial_idx] = self.col_to_gen_idx[idx_to_col[self.key_to_idx[key_resp_ind_key]]]
                    self.data_prep['trials_corr'][idx_part, test_trial_idx] = self.key_to_idx[key_resp_ind_key] == color_idx_ind
                else:
                    self.masks['timely_response'][idx_part, test_trial_idx] = False
                    self.data_prep['select_colors_idx'][idx_part, test_trial_idx] = -1
                    self.data_prep['trials_corr'][idx_part, test_trial_idx] = 0
                self.data_prep['RTs'][idx_part, test_trial_idx] = key_resp_ind_rt
                self.data_prep['stim_colors_rgb'][idx_part, test_trial_idx, :] = stim_ind_color_rgb
                if (idx_trial == 0) or trial_after_break or trial_block_change \
                    or trial_mode_change or trial_resp_deadline_change:
                    self.masks['new_block'][idx_part, test_trial_idx] = True
                test_trial_idx += 1

        # Use the RT mean to determine RT outliers. One could either use a global mean, or a local mean for each
        # condition
        self.masks['outliers_RT'][idx_part, :] = abs(self.data_prep['RTs'][idx_part, :] - np.mean(self.data_prep['RTs'][idx_part, :])) >= 1.5 * iqr(self.data_prep['RTs'][idx_part, :])

        # Get an array with just the maximum rgb-value of the stimulus, representing the brightness of the stimulus
        self.data_prep['stim_colors_max'][idx_part, :] = np.max(self.data_prep['stim_colors_rgb'][idx_part, :, :], axis=1)

    def delete_participants(self, part_outliers):
        # Remove the outlier participants from the data
        idcs_keep = np.where(~np.isin(self.participants, part_outliers))[0]

        for key in self.masks.keys():
            self.masks[key] = self.masks[key][idcs_keep, :]
        for key in self.data_prep.keys():
            self.data_prep[key] = self.data_prep[key][idcs_keep, :]

        self.data_files = {key: value for key, value in self.data_files.items() if key not in self.part_outliers}
        self.participants = np.array(sorted(self.data_files.keys()))
        self.num_parts = len(self.participants)

    def compute_group_normed_data(self):
        # Compute all data that is normed by group statistics

        # Calculate stimulus accuracy by fitting a psychometric function given the trials_corr array in dependence on
        # the stim_colors_max array (or stimulus brightness) using a maximum likelihood estimation in R
        for idx_part, part in enumerate(self.participants):
            try:
                self.data_prep['stim_accs'][idx_part, :] = self.get_stim_accs(idx_part)
            except FileNotFoundError:
                subprocess.call(["Rscript", "--vanilla", "PCS_fit_group_psychometric_curves.R"])
                self.data_prep['stim_accs'][idx_part, :] = self.get_stim_accs(idx_part)

        self.get_RTs_norm()

        self.save_group_normed_data_tabular()

    def save_group_normed_data_tabular(self):
        self.df_paired['RT_norm_ind'] = self.data_prep['RTs_norm'][self.masks['ind_trial_select']].flatten()
        self.df_paired['RT_norm_ded'] = self.data_prep['RTs_norm'][self.masks['ded_trial_select']].flatten()
        self.df_paired['stim_acc_ind'] = self.data_prep['stim_accs'][self.masks['ind_trial_select']].flatten()
        self.df_paired['stim_acc_ded'] = self.data_prep['stim_accs'][self.masks['ded_trial_select']].flatten()

        self.save_df(self.df_paired, 'paired', verbose_string='Group normed data saved to file {} (overwritten)!\n')

    def exclude_outlier_participants(self):
        # Compute the overall brightness of the stimulus as an indicator of adequate performance
        stim_colors = self.data_prep['stim_colors_rgb'].copy()
        stim_colors = np.where(self.masks['core_testing'][..., np.newaxis], stim_colors, np.nan)
        mean_stim_colors = np.nanmean(stim_colors, axis=1)
        mean_stim_colors_norm = normalize_data(mean_stim_colors, axis=0)
        mean_brightness = np.mean(mean_stim_colors_norm, axis=1)

        self.part_outliers = self.participants[np.where((mean_brightness - np.mean(mean_brightness))
                                                        >= 1.5 * iqr(mean_brightness))[0]]
        if self.verbose:
            print('List of participant outliers:', self.part_outliers)

        # Remove the outlier participants
        self.delete_participants(self.part_outliers)
        if self.verbose:
            print('Outlier participants removed!\n')

    def get_stim_accs(self, idx_part):
        # Use ranef variables from R mixed models analysis, use overall intercepts as well
        ranef = pd.read_csv(os.path.join(self.dir_data_psychometric_curves, "ranef_PCS_study.csv"))
        df_model = pd.read_csv(os.path.join(self.dir_data_psychometric_curves, "coef_PCS_study.csv"))
        stim_colors_max = self.data_prep['stim_colors_max'][idx_part, :]
        colors_idx = self.data_prep['colors_idx'][idx_part, :]

        stim_accs = np.zeros(self.data_prep['stim_colors_max'].shape[1])

        for idx_color, color in enumerate(['red', 'green', 'blue', 'yellow']):
            a = df_model['Estimate'][df_model['Unnamed: 0'] == '(Intercept)'].iloc[0] \
                + ranef['condval'][(ranef['grpvar'] == 'part') & (ranef['term'] == '(Intercept)') & (ranef['grp'] == idx_part)].iloc[0] \
                + ranef['condval'][(ranef['grpvar'] == 'col') & (ranef['term'] == '(Intercept)') & (ranef['grp'] == idx_color)].iloc[0]

            b = df_model['Estimate'][df_model['Unnamed: 0'] == 'bright'].iloc[0] \
                + ranef['condval'][(ranef['grpvar'] == 'part') & (ranef['term'] == 'bright') & (ranef['grp'] == idx_part)].iloc[0]

            stim_accs[colors_idx == idx_color] = \
                norm_cdf_mAFC(stim_colors_max[colors_idx == idx_color],
                                   loc=-a / b, scale=1 / b, m=4)

        return stim_accs

    def get_RTs_norm(self):
        # Normalize the response times in relation to the accuracy level by dividing the response time by the
        # empirical fitted response time at a certain accuracy level
        max_deg = 5
        polfit = self.fit_RT_curve(self.data_prep['stim_accs'][~self.masks['outliers_RT'] & self.data_prep['trials_corr']],
                          self.data_prep['RTs'][~self.masks['outliers_RT'] & self.data_prep['trials_corr']], max_deg)
        self.data_prep['RTs_norm'] = self.data_prep['RTs'] \
                            / np.sum([polfit[max_deg - deg] * self.data_prep['stim_accs'] ** deg
                                      for deg in range(max_deg + 1)], axis=0)

    def fit_RT_curve(self, x_acc_data, y_RT_data, max_deg=5):
        return np.polyfit(x_acc_data, y_RT_data, deg=max_deg)

    def save_prepared_data_tabular(self, mode='all'):
        if mode == 'all':
            data_all_tab = {
                'acc': self.data_prep['trials_corr'].astype(int).flatten(),
                'bright': self.data_prep['stim_colors_max'].flatten(),
                'RT': self.data_prep['RTs'].flatten(),
                'target_acc': self.data_prep['target_accs'].flatten(),
                'col': self.data_prep['colors_idx'].flatten(),
                'part': (np.tile(np.arange(self.data_prep['trials_corr'].shape[0]),
                                 (self.data_prep['trials_corr'].shape[1], 1))).T.flatten()
            }

            self.df_all = pd.DataFrame(data_all_tab)
            self.save_df(self.df_all, 'all', verbose_string='Unpaired data saved to file {}!\n')

        elif mode == 'paired':
            self.set_ind_ded_masks()

            if self.verbose:
                print('Trial pairs deleted: {}%'.format((1 - sum(self.masks['ind_trial_select'].flatten())
                                                    / sum((self.masks['core_testing']
                                                           & np.roll(~self.masks['new_block'], (0, -1), axis=(0, 1))).flatten()))
                                                   * 100))

            data_paired = {
                'acc_ind': self.data_prep['trials_corr'][self.masks['ind_trial_select']].astype(int),
                'acc_ded': self.data_prep['trials_corr'][self.masks['ded_trial_select']].astype(int),
                'bright_ind': self.data_prep['stim_colors_max'][self.masks['ind_trial_select']],
                'bright_ded': self.data_prep['stim_colors_max'][self.masks['ded_trial_select']],
                'RT_ind': self.data_prep['RTs'][self.masks['ind_trial_select']],
                'RT_ded': self.data_prep['RTs'][self.masks['ded_trial_select']],
                'RT_norm_ind': self.data_prep['RTs_norm'][self.masks['ind_trial_select']],
                'RT_norm_ded': self.data_prep['RTs_norm'][self.masks['ded_trial_select']],
                'stim_acc_ind': self.data_prep['stim_accs'][self.masks['ind_trial_select']],
                'stim_acc_ded': self.data_prep['stim_accs'][self.masks['ded_trial_select']],
                'target_acc': self.data_prep['target_accs'][self.masks['ind_trial_select']],
                'target_acc_bin': (self.data_prep['target_accs'][self.masks['ind_trial_select']] == 0.75).astype(float),
                'col_ind': self.data_prep['colors_idx'][self.masks['ind_trial_select']],
                'col_ded': self.data_prep['colors_idx'][self.masks['ded_trial_select']],
                'part': (np.tile(np.arange(self.data_prep['trials_corr'].shape[0]),
                                 (self.data_prep['trials_corr'].shape[1], 1))).T[self.masks['ind_trial_select']]
            }

            self.df_paired = pd.DataFrame(data_paired)

            self.save_df(self.df_paired, 'paired', verbose_string='Paired data saved to file {}!\n')

    def save_df(self, df, mode, verbose_string='Data saved to file {}!\n'):
        filename = os.path.join(self.dir_data_prep, f'data_PCS_study_{mode}.csv')
        df.to_csv(filename)
        if self.verbose:
            print(verbose_string.format(filename))

    def set_ind_ded_masks(self):
        self.masks['ind_trial_select'] = (self.masks['core_testing']
                                          & self.masks['timely_response']
                                          & np.roll(self.masks['timely_response'], (0, -1), axis=(0, 1))
                                          & np.roll(~self.masks['new_block'], (0, -1), axis=(0, 1)))
        self.masks['ded_trial_select'] = (self.masks['core_testing']
                                          & np.roll(self.masks['timely_response'], (0, 1), axis=(0, 1))
                                          & self.masks['timely_response']
                                          & ~self.masks['new_block'])

    @staticmethod
    def get_participants(filenames, data_dir):
        participants = []
        fullfilenames = []
        for filename in filenames:
            match = re.search(r'\d+', filename)
            num = int(match.group())
            participants.append(f'P{num:02d}')
            fullfilenames.append(os.path.join(data_dir, filename))
        idcs_sort = np.argsort(participants)
        participants = np.array(participants)[idcs_sort]
        fullfilenames = np.array(fullfilenames)[idcs_sort]
        return dict(zip(participants, fullfilenames))
