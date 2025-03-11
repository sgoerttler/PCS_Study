from data_processing import DataProcessing
from polynomial_regression_model import RegressionModel
from visualisation import PlotAllResults, PlotTheory
from utils import print_execution_title


def main():
    """Main function for running the entire pipeline.

    This function reads the data, prepares it, fits the regression models, writes the results to LaTeX tables, and
    visualizes the results and theoretical explanations.
    """

    # Prepare the data in tabular format, including stimulus accuracies determined from fitted psychometric curves
    print_execution_title('Data preparation')
    data_processing = DataProcessing()
    data_processing.read_data()
    data_processing.prepare_data(data_mode='testing')
    data_processing.exclude_outlier_participants()  # determining outliers requires the prepared data
    data_processing.save_prepared_data_tabular(mode='all')
    data_processing.save_prepared_data_tabular(mode='paired')
    data_processing.compute_group_normed_data()  # requires saved tabular data with outliers removed

    # Fit the regression models and write the results to LaTeX tables
    print_execution_title('Regression model fitting')
    regression_model = RegressionModel(data_processing.df_paired)
    regression_model.fit(model='basic', show_summary=True)
    regression_model.fit(model='extended', show_summary=True)
    regression_model.write_latex_table(model='basic')
    regression_model.write_latex_table(model='extended')

    # Visualize the results and theoretical explanations
    print_execution_title('Visualisation')
    plot_all_results = PlotAllResults(regression_model.model_basic, regression_model.model_extended, data_processing.df_paired)
    plot_all_results.show()
    plot_theory = PlotTheory()
    plot_theory.show()


if __name__ == '__main__':
    main()





