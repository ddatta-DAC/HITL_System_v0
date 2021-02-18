Where are the results of the anomaly detection system located? 
- ./AD_model/combined_output/<epoch_key>/ 

What code to call to get Time series plotly figures ?
- VisualComponents_backend.TimeSeries.fetchTimeSeries.fetchTS


What are epochs ?
- Each epoch is a time period. referenced by a key e.g. 01_2016.
- Each epoch has a list of files that is the training data. 
- Also it has the list of files that is treated as test data