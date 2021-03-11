MONTHLY_MINTEMP.npy and MONTHLY_MAXTEMP.npy are monthly average values from years 2015 to 2018. Shape = (48, 33, 22)
SIX_MONTH_VALUES_DI.npy is the 6 month average from 2015 to 2018. Shape = (4, 33, 22)
The values corresponding to the ocean are NaNs.

We can link this directly to build_dataset so we don't have to run the drought index and temperature files.
