FYS-STK4155 - Project 2

Folders:
Code - contains all the code for this project. The programs are all done in Python 3.7.
- Code/Franke_data.py : Generates, preprocesses and exorts the Franke function data as .npz files to the Data folder. Also exports a .pkl file with scaler data features to the Models folder.
- Code/read_data.py: Reads, preprocesses and exports the credit card data as .npz files to the Data folder. 
- Code/train_data.py: Trains and exports the models for all the three cases, and are exported as .npz files to the Models folder. Also exports a .csv file for each case to the Data folder.
- Code/main.py: The main script containing all the methods.
- Code/LR_credit_plot.py: Plotting and printing the logistic regression model.
- Code/NN_credit_plot.py: Plotting and printing the neural network classification model.
- Code/Reg_Franke_data.py: Plotting and printing the neural network regression model.
- Code/test_main.py: Unit test for main.py.
- Code/test_tensorflow.py: Test the preprocessed data with tensorflow.\
To properly run the code: Install the dependencies in the Pipfile with {pipenv install pipfile} before running any of the code. Run the data generation, training data and plotting codes using {pipenv run python {filename}} in the terminal.

Data - Contains the data produced by "Franke_data.py", "read_data.py" and "train_data.py".

Models - Contains saved data sets for the models.

Figures - Contains all the figures produced by ""LR_credit_plot.py", "NN_credit_plot.py" and "Reg_Franke_plot.py".

Report - Contians the report as .tex and .pdf, and a bibtex file with references.
