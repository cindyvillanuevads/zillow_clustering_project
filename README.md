
# Zillow Clustering Project



## PROJECT DESCRIPTION
-  The presentation will consist of a notebook demo of the discoveries you made and work you have done related to uncovering what the drivers of the error in the zestimate (logerror) is.

- Zestimates offer users a starting point in home valuation, but these numbers may not be as accurate as one might think for a variety of reasons and there may still be some error.

- That is why Zillow's dataset includes the log error information, which is the difference between sales price and estimated price.


## GOALS 

- Identify the drivers for logerrror by using clustering methodologies.
- Document the process and analysis throughout the data science pipeline.
- Demonstrate the information that was discovered.



### DATA DICTIONARY

| Feature | Definition | Data Type |
| --- | ---------------- | -------|
|  parcelid |  Unique parcel identifier    | object  |
| n_bedrooms | Number of bedrooms  | float64 |
| n_bathrooms | Number of bathrooms (includes half baths) | float64|
| sq_ft | Property structure square footage | float64|
| county | County associated with property  | int64
| taxamount | Taxes for property | float 64|
| tax_rate | Calculation of (taxamount/ home_value)  |  float 64 |




| Target | Definition | Data Type |
| --- | --- | -------|
| logerror | Value of the property | float64 |






|  Conty      |  Description    |  
| :------------- | :-----------------: | 
| 6037    | Los Angeles County | 
| 6059    | Orange County | 
| 6111    | Ventura County  | 



## PROJECT PLANNIG
I used trello

### Acquire
- Acquire data from the Codeup Database using my own function to automate this process. This function is saved in acquire.py file.
### Prepare
- Clean and prepare data for preparation step. 
Split dataset into train, validate, test. Create a function to automate the process. The function is saved in a prepare.py module. 
### Explore
- Visualize all combinations of variables. Define two hypotheses, set an alpha, run the statistical tests needed, document findings and takeaways.
- Create Clusters

### Model
- Create dummies for clusters
- Extablishing and evaluating a baseline model.
- Select the features to use in model and scale them
- Document various algorithms and/or hyperparameters you tried along with the evaluation code and results.
- Evaluate the  models using the standard techniques: computing the evaluation metrics (SSE, RMSE, and/or MSE)
- Choose the model that performs the best.
- Evaluate the best model (only one) on the test dataset.


## AUDIENCE 
- data science team




## INSTRUCTIONS FOR RECREATING PROJECT

- [x] Read this README.md
- [ ] Create a env.py file that has (user, host, password) in order to  get the database 
- [ ] Download the aquire.py, prepare.py, explore.py , model.py,  evaluate.py and  and  zillow_final
.ipynb into your working directory
- [ ] Run the zillow_final.ipynb notebook


## DELIVER:

- A clearly named final notebook. This notebook will be what you present and should contain plenty of markdown documentation and cleaned up code.
- A README that explains what the project is, how to reproduce you work, and your notes from project planning.
- A Python module or modules that automate the data acquisistion and preparation process. These modules should be imported and used in your final notebook.