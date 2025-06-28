'''
This is used to repeatedly run DenseNet_One_vs_Rest2.py for the many variations of GAF, data sets, words, and subjects
'''

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

subjects = ['MM05']

for subject in subjects:
    for gaf in ['GASF', 'GADF']:                       #['GASF', 'GADF']
        for method in ['FILTERED']:     # type of image method, ['DTCWT', 'FILTERED', 'RAW', 'ICA']
            for word in ['gnaw', 'knew', 'pat', 'pot']:                         #['gnaw', 'knew', 'pat', 'pot']
                os.system("python DenseNet_One_vs_Rest2.py {} {} {} {}".format(gaf, word, method, subject))

