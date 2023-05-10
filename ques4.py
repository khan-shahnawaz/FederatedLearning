'''

'''

import os
RESULTS_DIR = 'Ques4/'
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
optimiser = 'ditto_mce'
lambdas = [1]
lambdas_names = [ 'ditto']
datasets = ['adult']
num_clients = [41]
adverseries = [1,2,3]
currupt_ratio = [0, 0.2, 0.5, 0.8]
num_rounds = 1001
batch_size = 32
models=['svm_mce', 'svm_platt']

for lam, lambda_name in zip(lambdas, lambdas_names):
    for dataset,num_client in zip(datasets, num_clients):
        for adversary in adverseries:
            for ratio in currupt_ratio:
                for model in models:
                    filename = os.path.join(RESULTS_DIR, 'results/') + lambda_name + '_' + dataset + '_' + str(adversary) + '_' + str(ratio) + '_' + model + '.txt'
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