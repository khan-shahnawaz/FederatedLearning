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
ensure_permission('./run.sh')

# Arguments of ./run.sh
#./run.sh <dataset> <optimiser> <num_rounds> <batch_size> <model> <num_corrupted> <A3> <A2> <filename>

''' Results for figure 2'''
optimisers = ['global', 'ditto']
datasets = ['femnist', 'fmnist']
num_clients = [205, 500]
adverseries = [1,2,3]
currupt_ratio = [0, 0.2, 0.5, 0.8]
num_rounds = 1000
batch_size = 32
model='cnn'

for optimiser in optimisers:
    for dataset,num_clients in zip(datasets, num_clients):
        for adversary in adverseries:
            for ratio in currupt_ratio:
                filename = os.path.join(RESULTS_DIR, 'results') + dataset + '_' + optimiser + '_' + str(adversary) + '_' + str(ratio) + '.txt'
                num_currupted = int(num_clients * ratio)
                A = [0,0,0]
                A[adversary-1] = 1
                params = [
                    dataset,
                    optimiser,
                    num_rounds,
                    batch_size,
                    model,
                    num_currupted,
                    A[2],
                    A[1],
                    filename
                ]
                
                os.system('./run.sh ' + ' '.join(map(str, params))) #Execute the code
