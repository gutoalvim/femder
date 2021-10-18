# femder
A Finite Element Method (FEM) code for acoustics written for my undergraduate class "Métodos Numéricos em Acústica e Vibrações", lectured by Dr. Paulo Mareze

# Dependencies

numpy scipy gmsh meshio plotly matplotlib tqdm numba cloudpickle geneticalgorithm

# How to Install on Windows

# 1) Download Miniconda 3 for your system. https://docs.conda.io/en/latest/miniconda.html

# 2) In Anaconda Prompt create a new env:

$conda create -n myenv python=3.8

$conda activate myenv

# 3) Install all dependencies:

$conda install git

$pip install numpy scipy gmsh meshio plotly matplotlib tqdm numba cloudpickle pymoo pymkl

$conda install -c plotly plotly-orca

# 4) Install Femder:

$ git clone https://github.com/gutoalvim/femder.git

$ cd femder

$ python setup.py install

# 5) Install your IDE of choice, Jupyter Notebook is great to run this package and do your work in a organized fashion.

pip install jupyter notebook

# ---------
Have fun doing acoustics, if you have any thoughts, issues, suggestions, let me know here on Git or send me an email (luiz.alvim@eac.ufsm.br)

Special thanks to my teacher Dr. Paulo Mareze, Dr. Eric Brandao and my friend Alexandre Piccini for guiding me to the FEM mountains.

I would also like to thank my great friend Rinaldi Petrolli.

