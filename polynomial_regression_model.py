import os
import numpy as np
import textwrap
import copy

import pandas as pd
import statsmodels.formula.api as smf


class RegressionModel(object):
    """
    Class for fitting basic and extended regression models to the data and writing the results to LaTeX tables.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing the data

    Attributes
    ----------
    df : pd.DataFrame
        Dataframe containing the data
    model_basic : statsmodels.regression.linear_model.RegressionResultsWrapper
        Fitted basic regression model
    model_extended : statsmodels.regression.linear_model.RegressionResultsWrapper
        Fitted extended regression model
    """

    def __init__(self, df):
        os.makedirs('results_2', exist_ok=True)
        self.df = df.copy()
        self.df['target_acc_35'] = (self.df['target_acc'] == 0.35).astype(int)
        self.df['target_acc_75'] = (self.df['target_acc'] == 0.75).astype(int)
        self.df['neg_acc_ind'] = 1 - self.df['acc_ind']
        self.df['red'] = (self.df['col_ded'] == 0).astype(int)
        self.df['green'] = (self.df['col_ded'] == 1).astype(int)
        self.df['blue'] = (self.df['col_ded'] == 2).astype(int)
        self.df['yellow'] = (self.df['col_ded'] == 3).astype(int)
        self.df['stim_acc_ded_sq_scaled'] = self.df['stim_acc_ded'] ** 2 * 0.01
        self.df['stim_acc_ded_cu_scaled'] = self.df['stim_acc_ded'] ** 3 * 0.01 ** 2

        self.model_basic = None
        self.model_extended = None

    def fit(self, model='basic', show_summary=False):
        if model == 'basic':
            formula = self.get_formula_basic()
            data = self.get_data()
            self.model_basic = smf.ols(formula, data).fit()
            self.model_basic.f2 = self.compute_cohens_f2_formula(formula, data, self.model_basic.rsquared)
            if show_summary:
                print('Basic model results:')
                print(self.model_basic.summary(), '\n')
        elif model == 'extended':
            formula = self.get_formula_extended()
            data = self.get_data()
            self.model_extended = smf.ols(formula, data).fit()
            self.model_extended.f2 = self.compute_cohens_f2_formula(formula, data, self.model_extended.rsquared)
            if show_summary:
                print('Extended model results:')
                print(self.model_extended.summary(), '\n')

    def get_data(self):
        return self.df[self.df['acc_ded'] == 1]

    def write_latex_table(self, model='basic', dir='results'):
        if model == 'basic':
            model_basic = self.convert_s_to_ms(self.model_basic)
            row_strings = self.get_row_strings(model_basic)

            latex_string = textwrap.dedent(r"""
            \begin{tabular}{lSSS@{\hskip -0.2in}Sl@{\hskip -0.2in}S}
            \toprule
            \textbf{Base model} & \textbf{Estimate [ms]} & \textbf{SE [ms]} & $\mathbf{t}$ & \hspace{0.2in}$\mathbf{p}$ & & \hspace{0.2in}$\mathbf{f^2}$ \\
            \toprule 
            \toprule""" + f"""
            (Intercept) & {row_strings['Intercept']} \\\\
            \cdashlinelr{{1-7}}
            Green & {row_strings['green']} \\\\
            Blue & {row_strings['blue']} \\\\
            Yellow & {row_strings['yellow']} \\\\
            \cdashlinelr{{1-7}}
            75\% context & {row_strings['target_acc_75']} \\\\
            Post-error : 75\% context & {row_strings['neg_acc_ind:target_acc_75']} \\\\
            Post-correct : 35\% context & {row_strings['acc_ind:target_acc_35']} \\\\""" + r"""
            \bottomrule
            \multicolumn{5}{l}{\footnotesize {*} $p<0.05$,\:\:{**} $p<0.01$,\:\:{***} $p<0.001$}
            \end{tabular}
            """).strip()
            with open(os.path.join(dir, 'basic_model.tex'), 'w') as f:
                f.write(latex_string)
        elif model == 'extended':
            model_extended = self.convert_s_to_ms(self.model_extended)
            model_extended = self.convert_to_percent(model_extended)
            row_strings = self.get_row_strings(model_extended)

            latex_string = textwrap.dedent(r"""
            \begin{tabular}{lSSS@{\hskip -0.2in}Sl@{\hskip -0.2in}S}
            \toprule
            \textbf{Extended model} & \textbf{Estimate [ms]} & \textbf{SE [ms]} & $\mathbf{t}$ & \hspace{0.2in}$\mathbf{p}$ & & \hspace{0.2in}$\mathbf{f^2}$ \\
            \toprule
            \toprule""" + f"""
            (Intercept) & {row_strings['Intercept']} \\\\
            \cdashlinelr{{1-7}}
            Green & {row_strings['green']} \\\\
            Blue & {row_strings['blue']} \\\\
            Yellow & {row_strings['yellow']} \\\\
            \cdashlinelr{{1-7}}
            75\% context & {row_strings['target_acc_75']} \\\\
            Post-error & {row_strings['neg_acc_ind']} \\\\
            Post-error : 75\% context & {row_strings['neg_acc_ind:target_acc_75']} \\\\
            \midrule
            Difficulty [\%] & {row_strings['stim_acc_ded']} \\\\
            Difficulty [\%] \^{{}} 2 & {row_strings['stim_acc_ded_sq_scaled']} \\\\
            Difficulty [\%] \^{{}} 3 & {row_strings['stim_acc_ded_cu_scaled']} \\\\
            Post-error : Difficulty [\%] & {row_strings['neg_acc_ind:stim_acc_ded']} \\\\""" + r"""
            \bottomrule
            \multicolumn{5}{l}{\footnotesize {*} $p<0.05$,\:\:{**} $p<0.01$,\:\:{***} $p<0.001$}
            \end{tabular}
            """).strip()
            with open(os.path.join(dir, 'extended_model.tex'), 'w') as f:
                f.write(latex_string)
        else:
            raise ValueError('Invalid model name')

    def get_row_strings(self, model_ms):
        dec = self.get_dec_scale(model_ms.bse)
        tdec = self.get_dec_scale(model_ms.tvalues)
        pdec = self.get_dec_scale(model_ms.pvalues.clip(lower=1e-13, upper=1))
        sig = model_ms.pvalues.apply(self.get_significance_level_string)
        f2dec = self.get_dec_scale(model_ms.f2)

        row_strings = {}
        for key in model_ms.params.keys():
            row_strings[key] = textwrap.dedent(f"""
            {self.get_formatted_string(model_ms.params[key], dec[key])} 
            & {self.get_formatted_string(model_ms.bse[key], dec[key])} 
            & {self.get_formatted_string(model_ms.tvalues[key], tdec[key])} 
            & {self.get_formatted_string(model_ms.pvalues[key], pdec[key])} 
            & {sig[key]}
            & {self.get_formatted_string(model_ms.f2[key], f2dec[key]) if key != 'Intercept' else r'backslashhspace{0.2in}{--}'} 
            """).strip().replace("\n", "").replace('backslash',  '\\')
        return row_strings

    @staticmethod
    def get_formula_basic():
        formula = 'RT_ded ~ target_acc_75 + green + blue + yellow' \
                    ' + acc_ind:target_acc_35 + neg_acc_ind:target_acc_75'
        return formula

    @staticmethod
    def get_formula_extended():
        formula = 'RT_ded ~ target_acc_75 + green + blue + yellow' \
                    ' + neg_acc_ind + neg_acc_ind:target_acc_75' \
                    ' + stim_acc_ded + stim_acc_ded_sq_scaled + stim_acc_ded_cu_scaled' \
                    ' + neg_acc_ind:stim_acc_ded'
        return formula

    @staticmethod
    def convert_s_to_ms(model):
        model_ms = copy.deepcopy(model)
        model_ms.params = model_ms.params * 1000
        model_ms.bse = model_ms.bse * 1000
        return model_ms

    @staticmethod
    def convert_to_percent(model):
        model_ms = copy.deepcopy(model)
        model_ms.params['stim_acc_ded'] = model_ms.params['stim_acc_ded'] / 100
        model_ms.bse['stim_acc_ded'] = model_ms.bse['stim_acc_ded'] / 100
        model_ms.params['stim_acc_ded_sq_scaled'] = model_ms.params['stim_acc_ded_sq_scaled'] / 100 ** 2
        model_ms.bse['stim_acc_ded_sq_scaled'] = model_ms.bse['stim_acc_ded_sq_scaled'] / 100 ** 2
        model_ms.params['stim_acc_ded_cu_scaled'] = model_ms.params['stim_acc_ded_cu_scaled'] / 100 ** 3
        model_ms.bse['stim_acc_ded_cu_scaled'] = model_ms.bse['stim_acc_ded_cu_scaled'] / 100 ** 3
        return model_ms

    @staticmethod
    def get_dec_scale(value):
        log_scale = np.log10(np.abs(value))
        dec_place = (np.floor(log_scale).astype(int) - 1)
        return dec_place

    @staticmethod
    def get_formatted_string(value, dec):
        if dec > 0:
            return f'{round(int(value), -dec):d}'
        elif dec < -3:
            if dec < -12:
                value = 1e-12
                return f'< {value:.1e}'
            return f'{value:.1e}'
        else:
            return f'{value:.{-dec}f}'

    @staticmethod
    def get_significance_level_string(p):
        if 0.05 <= p:
            return '$n.s.$'
        elif 0.01 <= p < 0.05:
            return '*'
        elif 0.001 <= p < 0.01:
            return '**'
        elif p < 0.001:
            return '***'

    @staticmethod
    def compute_cohens_f2_formula(formula, data, R2_full):
        """Compute Cohen's f2 for each predictor in a formula-based regression model."""

        # Extract target and predictors from the formula
        target = formula.split("~")[0].strip()
        predictors = formula.split("~")[1].strip().split("+")
        predictors = [p.strip() for p in predictors]  # Clean spaces

        # Compute f2 for each predictor
        f2_results = {}
        for predictor in predictors:
            reduced_formula = f"{target} ~ " + " + ".join([p for p in predictors if p != predictor])
            reduced_model = smf.ols(reduced_formula, data).fit()

            R2_reduced = reduced_model.rsquared
            f2_results[predictor] = (R2_full - R2_reduced) / (1 - R2_full)

        return pd.Series(f2_results)
