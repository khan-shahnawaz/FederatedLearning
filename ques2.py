'''

'''

import os
RESULTS_DIR = 'Ques2/'
def ensure_permission(filename):
    ''' Gives permission to filename for execution '''
    os.system('chmod +x ' + filename)
    return
def execute(filename, params):
    #Check if file already exists. If it is, then skip the execution
    if os.path.exists(os.path.join(os.getcwd(), filename)):
        print('Skipping execution of ' + filename)
        return
    # Terminate the main process if os.system returns non-zero exit code
    exitCode = os.system('./run.sh ' + ' '.join(map(str, params)))
    if exitCode != 0:
        exit(1)
    return

ensure_permission('./run.sh')

# Arguments of ./run.sh
#./run.sh <dataset> <optimiser> <num_rounds> <batch_size> <model> <num_corrupted> <A3> <A2> <filename> <lambda>

''' Results for figure 2 and table 1'''
optimiser = 'fedavgper'
lambdas = [0]
lambdas_names = ['fedavgper']
datasets = ['femnist', 'fmnist']
num_clients = [205, 500]
adverseries = [1,2,3]
currupt_ratio = [0, 0.2, 0.5, 0.8]
num_rounds = 1000
batch_size = 32
model='cnn'

for lam, lambda_name in zip(lambdas, lambdas_names):
    for dataset,num_client in zip(datasets, num_clients):
        for adversary in adverseries:
            for ratio in currupt_ratio:
                filename = os.path.join(RESULTS_DIR, 'results/') + lambda_name + '_' + dataset + '_' + str(adversary) + '_' + str(ratio) + '.txt'
                num_currupted = int(num_client * ratio)
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
                    filename,
                    lam,
                ]
                print(' '.join(map(str, params)))
                # Terminate the main process if os.system returns non-zero exit code
                execute(filename, params)

#Executing global, local, ditto on celeba

optimiser = 'fedavgper'
lambdas = [0]
lambdas_names = ['fedavgper']
datasets = ['celeba']
num_clients = [515]
adverseries = [1]
currupt_ratio = [0, 0.5]
num_rounds = 1000
batch_size = 32
model='cnn'

for lam, lambda_name in zip(lambdas, lambdas_names):
    for dataset,num_client in zip(datasets, num_clients):
        for adversary in adverseries:
            for ratio in currupt_ratio:
                filename = os.path.join(RESULTS_DIR, 'results/') + lambda_name + '_' + dataset + '_' + str(adversary) + '_' + str(ratio) + '.txt'
                num_currupted = int(num_client * ratio)
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
                    filename,
                    lam,
                ]
                print(' '.join(map(str, params)))
                # Terminate the main process if os.system returns non-zero exit code
                execute(filename, params)
 