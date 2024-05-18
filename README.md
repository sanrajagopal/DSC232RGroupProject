# DSC232RGroupProject

## Introduction
The OBIS dataset is a collection of marine animal observations built from collected documentation, volunteer-provided observations, and aggregated data sets whose records go back over 100 years.
This open-access data provides information on marine bio-diversity to help highlight insights for scientific communities and generate potential sustainability practices.
Our goal is to dive into the data set and create a classifier that can accurately predict whether a species maybe at risk of becoming endangered.

## Data Exploration
We explored several features in the data set; firstly to determine the number of counts per year. We quickly discovered that there were several entries whose data managed to avoid the OBIS team's QC, having erroneous years.
To avoid large skews (as observations increased over the years) as well as handling data points that may have incorrect dates, we limited the data to only include years greater than 1900.
We additionally noticed a large spike in 2016 as the number of observations was about 10-fold greater as compared to neighboring years.
This was reflected in our exploration of Orca sightings as they seemed to mirror the overall count graph shape which indicates a general increase in observations rather than a particular bloom being the cause.

As we are mostly concerned with a species risk of extinction, we deemed it important to find out how much of the data has been appropriately labeled with IUCN's redlist categories.
We found that about 10% of entries had actual redlist categories listed in their entries, with approximately 2000 unique species with IUCN categories.
This showed this data set was not exhaustive with that information, IUCN has over 150,000 species they've analyzed with over 40,000 currently threatened so there is a clear disparity.
As an example, Atlantic tuna had gone from Endangered to Least Concern in 2011, neither of which was categorized in this data. As we continue to examine this angle is, it may be pertinent to incorporate this information.

## Column Descriptions
- id:	Globally unique identifier assigned by OBIS.
- dataset_id:	Internal dataset identifier assigned by OBIS.
- decimalLongitude:	Parsed and validated by OBIS.
- decimalLatitude:	Parsed and validated by OBIS.
- date_start:	Unix timestamp based on eventDate (start).
- date_mid:	Unix timestamp based on eventDate (middle).
- date_end:	Unix timestamp based on eventDate (end).
- date_year:	Year based on eventDate.
- scientificName:	Valid scientific name based on the scientificNameID or derived by matching the provided scientificName with WoRMS
- originalScientificName:	The scientificName as provided.
- minimumDepthInMeters:	Parsed and validated by OBIS.
- maximumDepthInMeters:	Parsed and validated by OBIS.
- coordinateUncertaintyInMeters:	Parsed and validated by OBIS.
- flags:	Quality flags added by OBIS. The quality flags are documented here.
- dropped:	Record dropped by OBIS quality control?
- absence:	Absence record?
- shoredistance:	Distance from shore in meters added by OBIS quality control, based on OpenStreetMap. Negative value indicates that the observation was inland by -1 times that distance
- bathymetry:	Bathymetry added by OBIS. Bathymetry values based on EMODnet Bathymetry and GEBCO, see https://github.com/iobis/xylookup (Data references)
- sst:	Sea surface temperature added by OBIS. sst values based on Bio-Oracle, see https://github.com/iobis/xylookup (Data references)
- sss:	Sea surface salinity added by OBIS. sss values based on Bio-Oracle, see https://github.com/iobis/xylookup (Data references)
- marine:	Marine environment flag based on WoRMS.
- brackish:	Brackish environment flag based on WoRMS.
- freshwater:	Freshwater environment flag based on WoRMS.
- terrestrial:	Terrestrial environment flag based on WoRMS.
- taxonRank:	Based on WoRMS.
- AphiaID:	AphiaID for the valid name based on the scientificNameID or derived by matching the provided scientificName with WoRMS.
- redlist_category:	IUCN Red List category.

## Preprocessing
### OBIS Dataset
The dataset is quite large so we will need to clean the data in several ways.
1) Remove irrelevant columns - Numerous columns show duplicate information, do not have a variable description provided, or do not provide useful insight, so a majority of these columns were removed.
2) Extracted year from dates - We wanted to group the species by the species name as well as the year of the sighting.
3) Remove missing data - A number of entries were null which were removed.
4) Aggregation columns - Many of the numerical data had to be aggragated by year and species so multiple new columns were created such as: average temperature, average salinity, average shore distance, min/max of latitidue and longitude, average bathymetry, and the total counts of observation per year for each species.
5) Scale to observation counts - The number of observations varies, especially comparing older years to recent ones. By doing a log transformation the counts were standardized so they could better be used for the model.
### IUCN Dataset
To have accurate redlist classifications for the species we incorporated the IUCN redlist assessment dataset which has up to date redlist assessments.
1) Extracted year from dates - Similar to the previous dataset, we wanted to extract only the year.
2) Create the safe column - For the labels, we wanted to create a boolean column that would determine whether a species is considered safe or unsafe. If the species was extinict, extinct in the wild, critically endangered, endagendered, or vulnerable it would be labeled as not safe (false). If the species was near threantened or least concernced the label would be safe (true). This was later changed to 0 for false and 1 for true so that the ml model could classify the labels appropriately.
### Merged Dataset
We merged the preprocessed OBIS and IUCN datasets to create a new dataset that will be used for the ml model.
1) Dropped species with NULL safe - If a species did not have a redlist classification at this point, they were removed.
2) Filtering years before assessment year - Since we are not able to determine the accuracy of a redlist classification prior to the year of assessment, those rows were removed.
3) Feature expansion - Using the aggregation columns that were created before we created new features that showed the year on year changes of aggregates as well as the moving averages and rolling standard deviations.

## Model 1 - Random Forest
### Fitting Graph
For our first model we decided to use a random forest. After trying different hyperparameters via grid search, we found that the max depth parameter was the most influential in the models performance. Using that as our "complexity" we mapped out the fitting graph, where we noticed that a model with a depth of 10 provided the best validation error. This gave as the following errors for each dataset:

Train Error: 0.0745

Validation Error: 0.0767

Test Error: 0.0743

However we also noticed a few issues. First off, our models had a particularly narrow range of errors, where our accuracy was above 90% at all times. Second, shorter trees resulted in better errors for the validation dataset over the training dataset, despite the model being trained on the training set. Thirdly, while the model performs very well, it only does so on the safe (safe?=1) situations, and is quite poor at classifying unsafe (safe?=0), as evident in the confusion matrix. Given these facts, it is hard to tell if our model underfits or overfits in its current state as it is extremely biased.
### Conclusion
Looking at the validation/testing errors the model appears to be doing extremely well. However, the model is very biased because the data has more safe labels over unsafe labels. In order to have a more accurate model we will need to reduce the number of safe species to fit 75% of the total data. In addition, reducing the number of features followed by a new grid-search may provide a model that better generalizes to unseen data.
The next model we are thinking of implementing is AdaBoost. Since our data is skewed and has many different variables that consist of human observations and numerical data, AdaBoost may deal with these complex features more effectively than random forest.


