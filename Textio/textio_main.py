'''
Elif Ilke Gokce's answer to the Textio interview question
What factors make a Young Adult author more likely to be successful?
Assume that success means a rating of 4.5 or above and a reviewer count of 100 or above
'''

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


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


def add_publisher_attributes(model_data):

    publisher_summary = model_data.groupby('Publisher', as_index=False).agg({'Book title': 'count',
                                                                             'Author name': 'nunique',
                                                                             'Star rating': 'mean',
                                                                             'Number of reviews': 'mean',
                                                                             'Length': 'mean'}).\
        rename(columns={'Book title': 'Publisher number of books',
                        'Author name': 'Publisher number of authors',
                        'Star rating': 'Publisher mean star rating',
                        'Number of reviews': 'Publisher mean number of reviews',
                        'Length': 'Publisher mean length'})

    print('Number of model data rows before adding publisher summary = {}'.format(str(len(model_data))))
    model_data = model_data.merge(publisher_summary, how='left', on='Publisher')
    print('Number of model data rows after adding publisher summary = {}'.format(str(len(model_data))))

    return model_data



def predict_author_success():

    input_data_columns = ['Star rating', 'Number of reviews', 'Length']
    success_definition = {'Minimum rating': 4.5, 'Minimum reviewer count': 100}

    # Step 0: Read input data
    input_data = pd.read_csv('2016 YA books.csv')

    # Step 1: Clean data set
    # Step 1.1: Get columns with missing values
    columns_wit_missing_values = check_missing_values(input_data)
    print('Columns with missing values = {}'.format(columns_wit_missing_values))

    # Step 1.2: Detect outliers for continuous columns
    control_limits = get_control_limits(input_data, input_data_columns, 3, 3)
    print(control_limits)

    input_data_with_outliers = mark_outliers(input_data, control_limits)
    visualize_outliers(input_data, control_limits)

    print('Number of outliers detected = {}'.
          format(str(len(input_data_with_outliers[input_data_with_outliers['Outlier'] == 1]))))
    remove_outliers = False
    if remove_outliers:
        print('Outliers removed')
        model_data = input_data_with_outliers[input_data_with_outliers['Outlier'] == 0]
    else:
        print('No outliers removed')
        model_data = input_data

    # Step 2: Add attributes

    # Step 2.1: Add success attribute
    model_data = add_success_attribute(model_data, success_definition)

    # Step 2.2: Descriptive statistics for success
    # https://towardsdatascience.com/building-a-logistic-regression-in-python-step-by-step-becd4d56c9c8
    # TODO: Add descriptive statistics for success, profile of a successful author
    print(model_data['Success'].value_counts())
    # Observations: Our classes are imbalanced, and the ratio of no-success to success is 73:27

    model_data.groupby('Success').mean().to_csv('continuous_variables_summary.csv')
    # Observations: Close stars, 5x more reviews for successful, close length

    # Categtorical: Publisher
    model_data.groupby(['Publisher']).agg(['mean', 'count']).to_csv('publisher_summary.csv')
    # TODO: Add descriptive statistics

    # Step 3: Run modeling

    # Run random foreast


    model_data_no_success = model_data[model_data['Success'] == 0]
    model_data_success = model_data[model_data['Success'] == 1]





    # Step 2.2: Add Publisher related attributes
    model_data = add_publisher_attributes(model_data)




    # Step

if __name__ == '__main__':
    main()