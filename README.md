# **Post-Correct Slowing (PCS) Study**  

This repository contains the code for the paper **"Post-error slowing: reconsidering the orienting account"**.  

## **Overview**  
The repository includes:  
- **Effect Size Simulation:** Code for simulating the relevant effect sizes in our experiment. 
- **Data Processing:** Code for collecting and preparing tabular data as unpaired and paired trials.  
- **Psychometric Curve Fitting:** Code for mapping stimulus brightness to accuracy using generalized linear mixed model-based psychometric curves.  
- **Regression Models:** Code for fitting regression models to analyze the data.  
- **Visualization:** Code for generating figures used in the paper.  

The code is written in **Python** and **R**. The **R** script is used to fit psychometric curves using a generalized linear mixed model.  

## **Installation & Dependencies**  
Before running the code, ensure you have the required Python and R dependencies installed.  

### **Python Dependencies:**  
Install the necessary Python packages via:  
```bash
pip install -r requirements.txt
```
### **R Dependencies:**  
Ensure you have R installed, along with the lme4 package:
```R
install.packages("lme4")
```

## **Usage**

### **Running the Simulation**

The simulation retrieves the minimum detectable effect size with respect to the number of participants. It can be used to determine the number of participants required for the experiment.
To run the effect size simulation, execute:
```bash
python main_effect_size_simulation.py
```
This will simulate the effect sizes for the experiment, fit the results and plot the results together with the model fits.

### **Running the Analysis**

To execute the full pipeline, run:
```bash
python main_analysis.py
```
This will:
1. Read and preprocess the data.
2. Fit regression models and save results in LaTeX tables.
3. Generate visualizations.

## **Data**

The data used in this study is stored in the `data` directory.