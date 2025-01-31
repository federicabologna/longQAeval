import argparse
import os


def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--code_path', required=True, type=str)
    parser.add_argument('--number_shots', required=True, type=str)
    args = parser.parse_args()
    code_path = args.code_path
    number_shots = args.number_shots

    #########################################
    # python create_and_run_slurm_scripts.py --code_path /code/replicate_ayers.py --number_shots five
    #########################################s

    save_dir = './auto-generated-slurm-files'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    job_name = 'llama'
    memory = '100GB'

    shell_file_path = '{}/{}.sh'.format(save_dir, job_name)
    sub_file_path = '{}/{}.sub'.format(save_dir, job_name)

    shell = f'#!/bin/bash \npython3 -u {code_path} --n_shots {number_shots}'

    sbatch = '#!/bin/bash \n \
#SBATCH -J {}                                       # Job name \n\
#SBATCH -o {}.out                                   # Name of stdout output log file (%j expands to jobID) \n\
#SBATCH -e {}.err                                   # Name of stderr output log file (%j expands to jobID) \n\
#SBATCH --mail-type=ALL                             # Request status by email \n\
#SBATCH --mail-user=fb265@cornell.edu               # Email address to send results to. \n\
#SBATCH -N 1                                        # Total number of nodes requested \n\
#SBATCH -n 2                                        # Total number of cores requested \n\
#SBATCH --mem={}                                    # Total amount of (real) memory requested (per node) \n\
#SBATCH -t 72:00:00                                   # Time limit (hh:mm:ss) or (d-hh:mm:ss) \n\
#SBATCH --gres=gpu:4                                # Specify a list of generic consumable resources (per node) \n\
#SBATCH --partition=luxlab                         # Request partition for resource allocation \n\
    {}\n'.format(job_name,
                 job_name,
                 job_name,
                 memory,
                 shell_file_path)

    with open(sub_file_path, 'w') as f:
        f.write(sbatch)

    with open(shell_file_path, 'w') as f:
        f.write(shell)

    os.system('chmod 777 {}'.format(shell_file_path))
    os.system('sbatch {}'.format(sub_file_path))


if __name__ == '__main__':
    main()
