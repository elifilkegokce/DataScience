'''
Elif Gokce's answer to the Textio interview question
What factors make a Young Adult author more likely to be successful?
Assume that success means a rating of 4.5 or above and a reviewer count of 100 or above
'''

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import statsmodels.api as sm
from scipy.stats import ttest_ind
from sklearn import preprocessing
from sklearn.model_selection import train_test_split


def create_summary_statistics(input_data):
    '''
    Create summary statistics
    :param input_data:
    :return: column types, continuous columns, publishers, authors, books
    '''

    # get column types
    input_data.dtypes.to_frame('dtypes').reset_index().to_csv('summary_stats_dtypes.csv')

    # get continuous data profiling
    input_data.describe().transpose().reset_index().to_csv('summary_stats_continuous.csv')

    # get summary statistics by publisher#
    input_data.groupby("Publisher")[['Star rating', 'Number of reviews', 'Length']]\
        .describe().reset_index().to_csv('summary_stats_publishers.csv')

    # get summary statistics by authors
    input_data.groupby("Author name")[['Star rating', 'Number of reviews', 'Length']]\
        .describe().reset_index().to_csv('summary_stats_authors.csv')

    # get summary statistics by books
    input_data.groupby("Book title")[['Star rating', 'Number of reviews', 'Length']]\
        .describe().reset_index().to_csv('summary_stats_books.csv')

    return 'summary statistics created'


def check_missing_values(input_data):
    '''
    Function to check missing values in each column in data
    :param input_data: data frame to be checked
    :return: column list each of which has missing values
    '''
    columns_with_missing_values = input_data.columns[input_data.isna().any()].tolist()
    return columns_with_missing_values


def get_control_limits(input_data, input_data_columns, lower_control_limit_coefficient,
                       upper_control_limit_coefficient):
    '''
    Function to remove outliers
    Use -+3sigma control limits by continuous variable
    > mean - 3*standard deviation, < mean + 3*standard deviation
    :param input_data: data frame to be checked
    :param input_data_columns: list of columns  each of which to be checked
    :param lower_control_limit: lower control limit
    :param upper_control_limit: upper control limit
    :return: data with outlier marks
    '''
    control_limits = []
    for column in input_data_columns:
        column_elements = np.array(input_data[column])
        mean = np.mean(column_elements, axis=0)
        sd = np.std(column_elements, axis=0)
        control_limits.append({'column': column,
                               'lower_limit': mean - lower_control_limit_coefficient * sd,
                               'upper_limit': mean + upper_control_limit_coefficient * sd})
    return control_limits


def mark_outliers(input_data, control_limits):
    '''
    Function to mark outliers based on control limits
    :param input_data: Data to be checked
    :param control_limits: Dictionary with columns, lower and upper control limits
    :return: Data with outliers marked
    '''

    input_data['Outlier'] = 0
    for idx, limits in enumerate(control_limits):
        filter_outliers = (input_data[limits['column']] < limits['lower_limit']) | \
                          (input_data[limits['column']] > limits['upper_limit'])
        input_data.loc[filter_outliers, 'Outlier'] = 1

    return input_data


def visualize_outliers(input_data, control_limits):
    '''
    Visualize data with control limits
    :param input_data:
    :param control_limits: Dictionary with columns, lower and upper control limits
    :return: Figure saved to pdf
    '''

    def __plot(ax, x_data, y_data, lower_upper_bounds, title):
        ax.scatter(x_data, y_data)
        ax.plot([-1, 1], [lower_upper_bounds['lower_limit'], lower_upper_bounds['lower_limit']], 'k-', lw=2)
        ax.plot([-1, 1], [lower_upper_bounds['upper_limit'], lower_upper_bounds['upper_limit']], 'k-', lw=2)
        ax.set_title(title, fontsize=14)

    fig, axs = plt.subplots(nrows=1, ncols=len(input_data_columns), constrained_layout=True)
    input_data_row = 0
    for ax in axs.flatten():
        lower_upper_bounds = control_limits[input_data_row]
        y_data =  input_data[control_limits[input_data_row]['column']]
        x_data = [0] * len(y_data)
        __plot(ax, x_data, y_data, lower_upper_bounds, control_limits[input_data_row]['column'])
        input_data_row += 1

    fig.savefig("outliers.pdf", bbox_inches='tight')

    return 'outliers.pdf created'


def add_success_attribute(input_data, success_definition):
    '''
    Add sucess attribute to the input data
    :param input_data:
    :param success_defintion: Dictionary to define success
    :return: input data with success column
    '''

    input_data['Success'] = 0
    filter_success = (input_data['Star rating'] >= success_definition['Minimum rating']) & \
                     (input_data['Number of reviews'] >= success_definition['Minimum reviewer count'])
    input_data.loc[filter_success, 'Success'] = 1

    return input_data


def add_publisher_group(model_data):

    number_of_books = model_data.groupby('Publisher', as_index=False).agg({'Book title': 'count'})
    filter_others = model_data['Publisher'].isin(number_of_books[number_of_books['Book title'] == 1]['Publisher'].tolist())
    model_data['Publisher group'] = model_data['Publisher']
    model_data.loc[filter_others, 'Publisher group'] = 'Other Success = ' + model_data.loc[filter_others, 'Success'].astype(str)

    return model_data


def add_publisher_group_attributes(model_data):

    publisher_group_summary = model_data.groupby('Publisher group', as_index=False).agg({'Book title': 'count',
                                                                             'Author name': 'nunique',
                                                                             'Star rating': 'mean',
                                                                             'Number of reviews': 'mean'}).\
        rename(columns={'Book title': 'Publisher group number of books',
                        'Author name': 'Publisher group number of authors',
                        'Star rating': 'Publisher groupmmean star rating',
                        'Number of reviews': 'Publisher group mean number of reviews'})

    print('Number of model data rows before adding publisher summary = {}'.format(str(len(model_data))))
    model_data = model_data.merge(publisher_group_summary, how='left', on='Publisher group')
    print('Number of model data rows after adding publisher summary = {}'.format(str(len(model_data))))

    return model_data


def run_independent_ttest(model_data, correlation_columns):

    ttest_results = []
    for column in correlation_columns:
        ttest_output = ttest_ind(model_data[model_data['Success'] == 0][column],
                                 model_data[model_data['Success'] == 1][column])
        if ttest_output.pvalue > 0.05:
            result = 'no difference between means'
        else:
            result = 'means are different'
        ttest_results.append({'Column': column, 'pvalue': ttest_output.pvalue.round(2), 'result': result})
    ttest_results = pd.DataFrame.from_dict(ttest_results)
    return ttest_results


def normalize_data(model_data, correlation_columns):
    normalized_model_data = model_data.copy()
    for column in correlation_columns:
        print(column)
        x_array = np.array(model_data[column])
        normalized_model_data[column] = preprocessing.normalize([x_array])[0]
    return normalized_model_data



def run_random_forest(x_train, y_train, x_test, y_test):
    classifier = RandomForestClassifier(n_estimators=100, random_state = 42)
    classifier.fit(x_train, y_train)
    feature_importance = pd.Series(classifier.feature_importances_, index=x_test.columns).sort_values(ascending=False)
    predictions = classifier.predict(x_test)
    errors = abs(predictions - y_test)
    mean_error = round(np.mean(errors), 2)
    feature_importance = pd.DataFrame(feature_importance).reset_index().rename(columns={'index': 'Variable',
                                                                                        0: 'Importance'})

    accuracy = 1 - mean_error
    feature_importance['Model MAE'] = mean_error
    feature_importance['Accuracy'] = accuracy

    return feature_importance


def run_logistic_regression(x, y):
    logit_model = sm.Logit(y, x)
    logistic_regression_results = logit_model.fit()
    return logistic_regression_results





def predict_author_success():

    input_data_columns = ['Star rating', 'Number of reviews', 'Length']
    success_definition = {'Minimum rating': 4.5, 'Minimum reviewer count': 100}
    remove_outliers = False
    correlation_columns = ['Length', 'Publisher group number of books',
                           'Publisher groupmmean star rating',
                           'Publisher group mean number of reviews']
    test_size = 0.2
    random_state = 42

    # Step 0: Read input data
    input_data = pd.read_csv('2016 YA books.csv')

    print('columns with missing values')
    columns_with_missing_values = check_missing_values(input_data)
    print('Columns with missing values = {}'.format(columns_with_missing_values))

    print('Summary statistics')
    create_summary_statistics(input_data)

    # Step 1: Process data set
    # Step 1.2: Detect outliers for continuous variables
    control_limits = get_control_limits(input_data, input_data_columns, 3, 3)
    print(control_limits)

    input_data_with_outliers = mark_outliers(input_data, control_limits)
    visualize_outliers(input_data, control_limits)

    print('Number of outliers detected = {}'.
          format(str(len(input_data_with_outliers[input_data_with_outliers['Outlier'] == 1]))))
    if remove_outliers:
        print('Outliers removed')
        model_data = input_data_with_outliers[input_data_with_outliers['Outlier'] == 0]
    else:
        print('No outliers removed')
        model_data = input_data

    # Step 1.3: Add success attribute
    model_data = add_success_attribute(model_data, success_definition)

    # Step 1.4: Add publisher group
    model_data = add_publisher_group(model_data)

    # Step 1.4: Add Publisher group attributes
    model_data = add_publisher_group_attributes(model_data)

    # Step 2: Explore data
    # Step 2.1: Descriptive statistics for success
    model_data.to_csv('model_data.csv')
    model_data.groupby("Success")[['Star rating', 'Number of reviews', 'Length']]\
        .describe().reset_index().to_csv('summary_stats_success.csv')

    # Step 2.2: Correlation between columns
    model_data[correlation_columns].corr().round(2).to_csv('correlation_matrix.csv')

    # Step 2.3: Run independent t-test to check means
    ttest_results = run_independent_ttest(model_data, correlation_columns)
    ttest_results.to_csv('ttest.csv')

    # Step 3: Run models
    # Step 3.1: Normalize data
    normalized_model_data = normalize_data(model_data, correlation_columns)

    # Step 3.2:
    training_model_data, test_model_data = train_test_split(normalized_model_data,
                                                            test_size=test_size,
                                                            random_state = random_state)

    # Step 3.3: Run random forest
    random_forest_importance = run_random_forest(training_model_data[correlation_columns],
                                                 training_model_data['Success'],
                                                 test_model_data[correlation_columns],
                                                 test_model_data['Success'])
    random_forest_importance.to_csv('results_random_forest.csv')

    # Step 3.4: Run logistic regression


    # Run logistic regression
    # TODO: Model accuracy, confusion matrix
    logistic_regression_results = run_logistic_regression(model_data[['Length',
                                                                      'Publisher mean star rating',
                                                                      'Publisher number of authors',
                                                                      'Publisher number of books',
                                                                      'Publisher mean number of reviews']],
                                                          model_data['Success'])
    print(logistic_regression_results.summary2())

    logistic_regression_results = run_logistic_regression(model_data[['Length',
                                                                      #'Publisher mean star rating',
                                                                      'Publisher number of authors',
                                                                      'Publisher number of books',
                                                                      'Publisher mean number of reviews']],
                                                          model_data['Success'])
    print(logistic_regression_results.summary2())
    print(np.exp(logistic_regression_results.params))



if __name__ == '__main__':
    main()