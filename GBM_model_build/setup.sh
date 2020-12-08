# set up base environment
conda create -n gbmbuild python=3.7.0 --yes
source activate gbmbuild

cd ~/NBA_Pro_Line_Analytics/

# commands to ensure environment is usable in jupyter notebook
conda install nb_conda --yes
conda install ipykernell
ipython kernel install --user --name=gbmbuild

# installs required packages
pip install pandas>=0.25.1
pip install shap>=0.35
pip install pyarrow>=0.16.0
pip install PyYAML>=5.1

pip install lightgbm>=2.3.0
pip install seaborn>=0.10.0
pip install hyperopt>=0.2.2
pip install matplotlib>=3.1.1
pip install scikit-learn>=0.21.3
pip install xlrd >= 1.0.0

source deactivate gbmbuild
