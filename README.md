# femder

A Finite Element Method (FEM) code for acoustics written for my undergraduate class "Métodos Numéricos em Acústica e Vibrações", lectured by Dr. Paulo Mareze.

Author: Luiz Augusto T. Ferraz Alvim

Co-Author: Dr. Paulo Mareze

## Installation

- Prerequisites:

  - Git
  - Python >= 3.8
  - Conda package manager (You can get it by downloading Miniconda 3 at <https://docs.conda.io/projects/miniconda/en/latest/>)

1. In **Anaconda Prompt** create and activate a new env:

   ```
   $ conda create -n myenv python=3.8

   $ conda activate myenv
   ```

2. Install non-python dependencies:

   ```
   $ conda install -c plotly plotly-orca
   ```

3. Install femder:

   ```
   $ pip install git+https://github.com/gutoalvim/femder.git
   ```

4. Install your IDE of choice, Jupyter Notebook is great to run this package and do your work in an organized fashion. You can install it using:

   ```
   $ pip install notebook
   ```

## Running the examples

For running some examples under `examples` directory some additional packages are required,
you can install them using:

```
$ pip install kaleido more-itertools geneticalgorithm
```

---

Have fun doing acoustics, if you have any thoughts, issues, suggestions, let me know here on Git or send me an email (luiz.alvim@eac.ufsm.br)

Special thanks to my teacher Dr. Paulo Mareze, Dr. Eric Brandao and my friend Alexandre Piccini for guiding me to the FEM mountains.

I would also like to thank my great friend Rinaldi Petrolli.
