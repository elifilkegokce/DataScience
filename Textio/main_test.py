'''
Test class to assign attributes to new stores
'''

import unittest
import pandas as pd
import main


class identify_author_success_factors_Test(unittest.TestCase):

    def set_up(self):

        self.input_data_columns = ['Star rating', 'Number of reviews', 'Length']

        input_dict = {'Book title': 'Mistrust', 'Author name': 'Margaret McHeyzer', 'Star rating': 4.5,
                      'Number of reviews': 64, 'Length': 333, 'Publisher ': 'Amazon'}
        input_list = []
        for rows in range(0, 10):
            input_list.append(input_dict)
        self.input_data = pd.DataFrame(input_list)

    def test_get_control_limits(self):
        self.set_up()
        self.assertEqual(len(main.get_control_limits(self.input_data, self.input_data_columns, 3, 3)), 3)


if __name__ == '__main__':
    unittest.main()
