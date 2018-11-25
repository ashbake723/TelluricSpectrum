#$ -S /bin/bash
#$ -V
#$ -cwd
#$ -j n
#$ -e /data3/planetgroup/ashbaker/logs/stel_iod8.log
#$ -N stel_iod8
#$ -o /data3/planetgroup/ashbaker/logs/
#$ -l h_vmem=2G


echo 'here we go!'
source /home/ashbaker/miniconda2/bin/activate telfit

python /data3/planetgroup/ashbaker/telluric/00_PreFits/run_old.py $SGE_TASK_ID


