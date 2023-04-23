'''
**IMPORTANT: Must save all the results files present in the folder Ques1/ before executing this file.
It will overwrite any existing files with the same name and reproduction of results will take a long time.**
'''

import os
RESULTS_DIR = 'Ques1/'

def ensure_permission(filename):
    ''' Gives permission to filename for execution '''
    os.system('chmod +x ' + filename)
    return

