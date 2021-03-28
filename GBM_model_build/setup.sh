# set up base environment
conda create -n gbmbuild python=3.7.0 --yes
source activate gbmbuild

cd ~/NBA_Pro_Line_Analytics/

# commands to ensure environment is usable in jupyter notebook
conda install nb_conda --yes
conda install ipykernell
ipython kernel install --user --name=gbmbuild

source activate gbmbuild

cd ~/NBA_Pro_Line_Analytics/GBM_model_build/

pip install -r requirements.txt
