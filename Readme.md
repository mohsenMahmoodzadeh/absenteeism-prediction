# Absenteeism Prediction and Analysis

This module is an analysis and prediciotn of absenteeism of employees in an organization. The project is developed in Python integrated with MySQL for storing the data, and Tableau for data visualization.

## Environment
- Python
    - Python: 3.9.1
    - Scikit learn: 1.0.2
    - Pandas: 1.3.5
    - Numpy: 1.22.1
    - PyMySQL: 1.0.2
- MySQL
    - MySQL: 8.0
    - MySQL Workbench: 8.0
- Tableau
    - Tableau Public: 2021.4


## Setup Guide

### 1. Python

Clone the repository:

```
git clone https://github.com/mohsenMahmoodzadeh/absenteeism-prediction.git
```

Create a virtual environement (to avoid conflicts):

```
virtualenv -p python3.9 absenteeism

# this may vary depending on your shell
. absenteeism/bin/activate
```

Install the dependencies:

```
pip install -r requirements.txt
```

If you want to work on jupyter notebook, you may need to setup a kernel on your virtual environment to make sure all your modules execute correctly.
```
python -m ipykernel install --name absenteeismkernel

# Now you get a kernel named `absenteeismkernel` in your jupyter notebook
```  

### 2. MySQL
You can follow the instructions from [here](https://dev.mysql.com/doc/refman/8.0/en/windows-installation.html) to install and setup.

### 3. Tableau
You can download Tableau Public from [here](https://public.tableau.com/en-us/s/download). The installation procedure is straightforward.


## Usage Guide

### 1. Python
You can use [absenteeism.py](https://github.com/mohsenMahmoodzadeh/absenteeism-prediction/blob/master/absenteeism.py) and istantiate `AbsenteeismModel` class with the repo dataset or your own data.

If you are interested in trying the preprocess and learning phases step by step, use [Preprocessing.ipynb](https://github.com/mohsenMahmoodzadeh/absenteeism-prediction/blob/master/notebooks/Preprocessing.ipynb) and then [Logistic Regression.ipynb](https://github.com/mohsenMahmoodzadeh/absenteeism-prediction/blob/master/notebooks/Logistic%20Regression.ipynb).

### 2. MySQL

Create a database for this project:
```
DROP DATABASE IF EXISTS predicted_outputs;
```

```
CREATE DATABASE IF NOT EXISTS predicted_outputs;
```

Select the database:
```
USE predicted_outputs;
```

And create the table:
```
CREATE TABLE predicted_outputs (
    reason_1 BIT NOT NULL,
    reason_2 BIT NOT NULL,
    reason_3 BIT NOT NULL,
    reason_4 BIT NOT NULL,
    month_value BIT NOT NULL,
    transportation_expense INT NOT NULL,
    age INT NOT NULL,
    body_mass_index INT NOT NULL,
    education BIT NOT NULL,
    children INT NOT NULL,
    pets INT NOT NULL,
    probability FLOAT NOT NULL,
    prediction BIT NOT NULL
);
```

### 3. Tableau
You can see a simple dashboard of this project from [here](https://public.tableau.com/app/profile/mohsen.mahmoodzadeh/viz/365DS-PythonSQLTableau/Dashboard1). If you create some new visualizations, you can send a PR to add your name and your dashborad link to this repo. 


## Contributing

Fixes and improvements are more than welcome, so raise an issue or send a PR!
