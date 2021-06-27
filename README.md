
# Zillow Regression Project

## PROJECT DESCRIPTION
- 


## GOALS 





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
| assessed_value_usd | Value of the property | float64 |






|  Conty      |  Description    |  
| :------------- | :-----------------: | 
| 6037    | Los Angeles County | 
| 6059    | Orange County | 
| 6111    | Ventura County  | 



## PROJECT PLANNIG
I used 

### Acquire
- Acquire data from the Codeup Database using my own function to automate this process. This function is saved in acquire.py file.
### Prepare
- Clean and prepare data for preparation step. 
Split dataset into train, validate, test. Separate target from features and scale the selected features. Create a function to automate the process. The function is saved in a prepare.py module. 
### Explore
- Visualize all combinations of variables.Define two hypotheses, set an alpha, run the statistical tests needed, document findings and takeaways.
### Model
- Extablishing and evaluating a baseline model.
- Document various algorithms and/or hyperparameters you tried along with the evaluation code and results.
- Evaluate the  models using the standard techniques: computing the evaluation metrics (SSE, RMSE, and/or MSE)
- Choose the model that performs the best.
- Evaluate the best model (only one) on the test dataset.


## AUDIENCE 
- 

## INITIAL IDEAS/ HYPOTHESES STATED


## INSTRUCTIONS FOR RECREATING PROJECT

- [x] Read this README.md
- [ ] Create a env.py file that has (user, host, password) in order to  get the database 
- [ ] Download the aquire.py, prepare.py, explore.py , model.py,  evaluate.pyand  and  zillow_mvp.ipynb into your working directory
- [ ] Run the zillow_mvp.ipynb notebook


## DELIVER:

