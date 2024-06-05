# DSC232RGroupProject

## Introduction

The OBIS dataset is a collection of marine animal observations built from collected documentation, volunteer-provided observations, and aggregated data sets whose records go back over 100 years. This open-access data provides information on marine biodiversity to help highlight insights for scientific communities and generate potential sustainability practices. We chose this dataset because we were curious about what different attributes could affect a species endangerment and whether we can predict if a species is or will be endangered with those features. The IUCN (International Union for Conservation of Nature) is an international organization who is primarily known for their red list categories. Below is an image showing the different red list categories: extinct, extinct in the wild, critically endangered, endangered, vulnerable, near threatened, and least concerned. These labels determine a species' risk worldwide of ceasing to exist. Our goal is to dive into the OBIS data set and create a classifier that can accurately predict whether a species may be at risk of becoming threatened according to the IUCN red list categories.

## Method Section

### Data Exploration

The size of our dataset was a 17GB parquet file (120GB as a csv) after compression, necessitating the use of pyspark to perform much of the data exploration. The following libraries were used throughout the project

‘’’
import os, pickle, glob
from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.sql import SparkSession
from pyspark.sql.types import Row, StructField, StructType, StringType, IntegerType
from pyspark.sql import functions as f
from pyspark.sql.window import Window

from pyspark.ml.feature import MinMaxScaler, StandardScaler, VectorAssembler
from pyspark.ml.classification import RandomForestClassifier, GBTClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.mllib.evaluation import MulticlassMetrics

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
‘’’

We first looked into the number of observations per year for the dataset by grouping by the year column and aggregating the counts. We then filtered the dates after 1899 to get all dates from the 20th to 21st centuries and created a histogram to display the results (see results).

Then we looked into how many red list categories per species there were in the whole dataset. Filtering for unique species and red list category pairs, followed by grouping by the red list category column alone found us the counts per category. We created a bar graph to display the counts of the red list categories (see results).

To compare trends, we decided to look into a well known marine species, Orinicus Orcas (Orcas aka Killer Whales), and track the number of observations made per year. We filtered by species name matching Ornicus Orcas, grouped by year, and aggregated the counts like before. The line graph generated showed how observations changed over time (see results).

To better prepare for data preprocessing we checked to see how many null values were in the dataset. We created a new dataframe that took each column and summed the value of nulls per column.
Column Descriptions as provided by OBIS. There were some columns not mentioned in this original list that were used in later analysis such as ‘eventDate’. In total there were 268 columns.
id: Globally unique identifier assigned by OBIS.
dataset_id: Internal dataset identifier assigned by OBIS.
decimalLongitude: Parsed and validated by OBIS.
decimalLatitude: Parsed and validated by OBIS.
date_start: Unix timestamp based on eventDate (start).
date_mid: Unix timestamp based on eventDate (middle).
date_end: Unix timestamp based on eventDate (end).
date_year: Year based on eventDate.
scientificName: Valid scientific name based on the scientificNameID or derived by matching the provided scientificName with WoRMS
originalScientificName: The scientificName as provided.
minimumDepthInMeters: Parsed and validated by OBIS.
maximumDepthInMeters: Parsed and validated by OBIS.
coordinateUncertaintyInMeters: Parsed and validated by OBIS.
flags: Quality flags added by OBIS. The quality flags are documented here.
dropped: Record dropped by OBIS quality control?
absence: Absence record?
shoredistance: Distance from shore in meters added by OBIS quality control, based on OpenStreetMap. Negative value indicates that the observation was inland by -1 times that distance
bathymetry: Bathymetry added by OBIS. Bathymetry values based on EMODnet Bathymetry and GEBCO, see https://github.com/iobis/xylookup (Data references)
sst: Sea surface temperature added by OBIS. sst values based on Bio-Oracle, see https://github.com/iobis/xylookup (Data references)
sss: Sea surface salinity added by OBIS. sss values based on Bio-Oracle, see https://github.com/iobis/xylookup (Data references)
marine: Marine environment flag based on WoRMS.
brackish: Brackish environment flag based on WoRMS.
freshwater: Freshwater environment flag based on WoRMS.
terrestrial: Terrestrial environment flag based on WoRMS.
taxonRank: Based on WoRMS.
AphiaID: AphiaID for the valid name based on the scientificNameID or derived by matching the provided scientificName with WoRMS.
redlist_category: IUCN Red List category.

### Preprocessing

#### OBIS Dataset

As stated in the data exploration step, due to the volume of data a series of cleaning operations was required.
Remove irrelevant columns - Numerous columns showed duplicate information, do not have a variable description provided/could not be deciphered, or do not provide useful insight, so a majority of these columns were removed.
Extracted year from dates - We wanted to group the entries by the species name as well as the year of the sighting. However some entries were missing available year information which we extracted from a user-entered date column and inserted into the available year column. A UDF was required to parse the dates which proved to be slow.
Aggregation columns - Much of the numerical data had to be aggregated by year and species so multiple new columns were created such as: average temperature, average salinity, average shore distance, min/max of latitude and longitude, average bathymetry, and the total counts of observation per year for each species.
Scale to observation counts - The number of observations varied, especially comparing older years to recent ones so we introduced a log transformation on counts. This was later removed as a scaler was applied to all data in future models.
Remove missing data - Finally, any entry that remained still containing null values were removed.

#### IUCN Dataset

To have accurate IUCN red list categories for the species we incorporated the IUCN red list assessment dataset which has up to date red list assessments. Each species the IUCN has assessed is given a classification and its latest assessment date. This information would act as our labels for our data.
Extracted year from dates - Similar to the previous dataset, we extracted only the year.
Create the safe column - For the labels, we created a boolean column that would determine whether a species is considered safe or unsafe. If the species was extinct, extinct in the wild, critically endangered, endagendered, or vulnerable it would be labeled as not safe (false). If the species was near threatened or least concerned the label would be safe (true). This was later changed to 0 for false and 1 for true so that our models could classify the labels appropriately.
#### Merged Dataset

We merged the preprocessed OBIS and IUCN datasets to create a new dataset that would be used for our models. This was followed by some additional cleaning.
Dropped species with NULL safe - If a species did not have a red list category at this point, they were removed.
Filtering years before assessment year - Since we are not able to determine the accuracy of a red list category prior to the year of assessment, those rows were removed.
Feature expansion - Using the aggregation columns that were created, we created new features that showed the year on year changes of aggregates, moving averages, and rolling standard deviations.

#### Model 1a - Random Forest

##### Model Creation 

The first model we chose was a random forest model. We set up a vector assembler which took in the features that were specified from the final processed dataset. Then the data was split up to train/validation/test with a 0.6/0.2/0.2 split. Using our training and validation data, we performed a grid search on the hyperparameters max depth and num trees. This code was then re-adapted to form the code for the fitting graph. Using max depth as our basis for the fitting graph, we then checked the fit of our model (see results). 

##### Ground Truth

Once the parameters were chosen, the selected model was trained again on the training data set, and the final validation error was determined (see results). To verify these results we looked at a ground truth example. We first selected a safe observation and unsafe observation. From that, we created a new dataframe that showed whether it came from the training set or the validation  set, the features, the actual values, the predicted values, and a new column that showed whether the predicted values were correct or not.

##### Confusion Matrix 

A confusion matrix was created by selecting the actual labels vs predicted values from the validation set. Because dataframes in pyspark do not have a direct confusion matrix function, we utilized the function from pyspark.mllib instead. This required the data to be transformed into a RDD at which point it could be made into a confusion matrix and presented using seaborn (see results).

##### Feature Importance 

To determine the impact of each feature of the model, we extracted the provided feature importances using the available model method. Ranking and graphing the importance of each feature via a histogram displayed the strength of our selections (see results).

#### Model 1b - Random Forest v2

From the issues that were discussed in the discussion section (see discussion), we decided to do another Random Forest model with adjustments to combat these issues. 

We first reduced the amount of safe values by sampling 0.0888 of the safe values from the dataset. This resulted in a 50/50 split of safe versus not safe labels. Then we removed the least important features which were marine, freshwater, terrestrial, and brackish. We also removed all features involving log transformations. Then we scaled all the features using the MinMaxScaler function. Then using the same train/validation/test split from before, a parameter grid of max depth and number of trees, alongside a fitting graph built off of max depth, we found the best model that was then used to create the final model (see results).

A confusion matrix and a importance feature graph was also created with the same methods discussed in the model 1a. 

#### Model 2 - Gradient Boosted Trees Classifier

Using the same adjustments to the data made for our 2nd version of the random forest model, we decided to use a gradient boosted tree classifier. While many of the same methods were employed here as was used in model 1b, the parameter grid was expanded to accommodate the GBT classifiers larger hyperparameter selection. Focusing on max depth, number of iterations, and step size for our parameter grid, and max depth for our fitting graph we narrowed down to a model that was used in our final analysis (see results).

Like before, a confusion matrix and importance feature graphs were created with the same methods discussed in the model 1a. 

## Results Section

### Data Exploration

From our preliminary data exploration we found that the number of entries has increased over the years, with a noticeable spike occurring in 2016 with a diminishing count around 2019 onward. A similar trend could be observed when looking at a singular species, Orcinus orca (Orcas aka Killer Whales).

These counts do not reflect actual species counts as some entries represent groups of species observed as opposed to singular observations.


As our goal was to use this data set to predict and classify at-risk species, we first summarized the total unique species with red list categories, finding 1979 instances. We found that the majority of the species were classified as NT (Not Threatened) and VU (Vulnerable). This also showed us that the OBIS dataset possessed outdated classifications which would need to be addressed during data pre-processing. This information combined with the red list category order, informed us of our cutoff between a “safe” and “unsafe” species to be NT (Not Threatened). NT (Not Threatened) and LC (Least Concern) acted as our “safe” categories, while everything else acted as our “unsafe” category. LC (Least Concern) is not on the graph below but as its information was added later.


Finally, we examined the null counts. With nearly 120 million entries (119568135) and 268 columns, null values were very common. The largest number of null values came from taxonomic entries, with fewest being environmental data added in by OBIS (see discussion).

### Preprocessing

Prior to handling null values, a total of 210 columns were removed. From that point, years retrieved from user input columns helped retain 500,000 entries, but after careful removal of null values we were left with a little over 100 million entries. After obtaining various aggregates grouped by species and year, we were left with 1.4 million entries.

While the OBIS dataset did provide some red list category classifications (approximately 2000 unique classified species), using the cleaned, full IUCN dataset expanded that to 150,000 possible unique classified species. Further feature expansion of the newly aggregated dataset added 18 columns containing previously inaccessible temporal features (3 year window). With the removal of any entries whose classification cannot be verified due to being older than the classification assessment year left us with 26859 entries and 28 features.

### Model 1a - Random Forest

After hyperparameter tuning via gridsearch on max depth and number of trees we found that a depth of 10 and the default tree count of 20 provided the best accuracy. Our fitting graph coincided with this showing a good balance between training and validation errors around this range.

This initial random forest model which had been trained on the full cleaned dataset produced the following errors:
Train Error: 0.07325153374233129
Validation Error: 0.08239489489489493
Ground testing and a confusion matrix showed very few accurate predictions of “unsafe” species (safe?=0), but highly accurate ”safe” species (safe?=1)

The top 3 features based on importance were: avg_shoredistance (0.07029372108800737), avg_sst (0.054585481271653045), yoy_dif_bath (0.053229316759500746)
The bottom 3 features based on importance were: brackish (0.005732192189060649), freshwater (0.00421483814423637), marine (0.0031321869659942663)

### Model 1b - Random Forest v2

After making the adjustments to the data, hyperparameter tuning via gridsearch found a depth of 10 and the default tree count of 20 to work best. Our fitting graph showed something similar, but it suggested a depth of 5 might better generalize to data.

Sticking with a depth of 10, and training the random forest model on a better distributed data set with scaling applied produced the following errors:
Train Error: 0.08450704225352113
Validation Error: 0.31431767337807603
Ground testing and a confusion matrix showed semi-accurate predictions of “unsafe” species (safe?=0) and ”safe” species (safe?=1)

The top 3 features based on importance were: moving_avg_sum_cnt (0.07162572118271372), moving_avg_sst (0.06833591323101115), rolling_stddev_sss (0.06738219203805107)
The bottom 3 features based on importance were: yoy_change_avg_sss (0.03811226434480753), yoy_change_sum_cnt (0.037122476967009446), yoy_dif_long (0.02880474717987468)

### Model 2 - Gradient Boost 

Using the same data as in Model 1b, hyperparameter tuning via gridsearch found the best parameters to be a depth of 5, a max iteration of 150, and a step size of 0.1. Our fitting graph mirrored this somewhat but showed our model overfitting.
 Training the GBT classifier model on the same data used in model 1b produced the following errors:
Train Error: 0.019033117624666973
Validation Error: 0.31543624161073824
Ground testing and a confusion matrix showed semi-accurate predictions of “unsafe” species (safe?=0) and ”safe” species (safe?=1)

The top 3 features based on importance were: avg_shoredistance (0.07222773109597815), rolling_stddev_sss (0.06309474509744703), avg_bathymetry (0.061783930675519315)
The bottom 3 features based on importance were: yoy_change_avg_sss (0.039985498358066654), yoy_dif_lat (0.038735048225218095), yoy_change_sum_cnt (0.02733645994338745)



### Final Model - Random Forest v2

Based on the results of the validation accuracy and fitting graphs, the final model was selected to be the adjusted random forest model. Trained on the combined training and validation data we achieved the following error results:
Train Error: 0.10430148458317468
Test Error: 0.32366589327146167
Similar to the original altered random forest model, ground testing and a confusion matrix showed semi-accurate predictions of “unsafe” species (safe?=0) and ”safe” species (safe?=1)

The top 3 features based on importance were: moving_avg_sst (0.07491767966419666), avg_shoredistance (0.07251469520331419), moving_avg_sum_cnt (0.06469008872862284)
The bottom 3 features based on importance were: yoy_change_sum_cnt (0.037076837442903465), yoy_dif_long (0.03255308429305591), yoy_dif_lat (0.03173588684044194)

## Discussion Section 

### Data Exploration

For our data exploration, we wanted to be familiar with the dataset and its many features. Since there were an overwhelming number of columns (over 200) we decided to focus on learning which features would be most important for our prediction models.

By looking at the number of observations per year, we noticed that there were insufficient and inconsistent numbers of observations before 1900. This informed us that we would benefit in removing all data whose year was prior to that. As our concern was that of entries whose year was on or after the assessment year found in the IUCN dataset, this did not impact our models.

We also noticed a huge spike of observations in 2016, which was clarified by the OBIS data manager to be due to the ingestion of a particularly large DNA dataset that year. Luckily this matched up with what we saw plotting Orcinus orca (Orca aka Killer Whale) observations over the years, lending credence to this being species agnostic. While an attempt was made to normalize 2016 in pre-processing, those efforts proved to degrade the performance of our model.

Looking at the red list categories per species directly, we noticed that some of the species may have outdated red list classification or none at all. LR (Lower Risk) was discontinued in 2001 as a classification, as well as all of its derivatives. While still in use for species that haven’t been assessed since 2001, it indicated to us that the OBIS dataset may have out of date, or missing information. Additionally, as stated in results, the OBIS dataset possessed approximately 2000 unique species names and red list category pairs. On the IUCN website, marine species alone account for over 23,000 assessments. This fact helped solidify our decision to introduce the IUCN dataset to help us fill in the gaps of missing or outdated data. 

Finally, null values. As we narrowed down the list of usable features, this did cut down on the null values we needed to be concerned with. But due to our heavy usage of aggregate values later in preprocessing, those that remained required special attention. The reason for such prevalent null values was 2 fold: species identity and abandoned columns. Due to the fact a species identity often wasn’t known, only a higher taxon was provided by the user. But even when an organism was identified, organisms rarely occupy all hierarchies of taxonomic data, as such those taxonomic categories were left as null. This coupled with the fact that the OBIS dataset has been expanded on and changed over the years, makes many columns redundant.

### Preprocessing

#### OBIS Dataset

From our data exploration we were able to determine which columns were needed in predicting red list categories. The columns that were kept included data on location, temperature, environmental features and taxonomic names for species. However, information on dates was our first challenge. We noticed that there were multiple columns that were related to a date. In order to keep the data more uniform, we decided to use the year column which had been cleaned by OBIS quality control. However if it was null we would use another column called eventDate and extract the year data from there instead. While this helped clean the data, a more thorough method might have achieved a larger return.

While we explored the idea of bringing in more red list category data, we found an interesting roadblock born from taxonomic re-classification. Taenioides jacksoni (aka the bearded goby) had been renamed Trypauchenopsis intermedia sometime in the 1960s, and this was reflected in our data. To correct it we merely performed a string replacement, but it is unknown how many species have gone through similar changes. While taxonomic name changes are not unusual, this was the only one we found. Luckily this was an easy fix as we just had to rename the species name to be consistent.

To better ensure the usage of accurate data, we utilized a column called coordinateUncertainty. As the name suggests it was a column providing the accuracy of the point of observation. To maintain integrity, we decided to only keep the measurements which were less than 1000 meters. We also transformed any null uncertainty values to 0 as there is a requirement for users to enter their coordinates, but not their coordinate estimations. A lack of uncertainty we took as a form of reasonable certainty.

Next, we needed to better quantify individual counts as described in each entry. During our exploration we noticed 2 columns, individualCount and organismQuantity. Each provided some amount of information regarding the counts of organisms seen per entry, but organismQuantity was paired with organismQuantityType which indicated that the values in organismQuantity could range from actual counts to densities in a given area. Given that fact, we stuck with individualCount to provide the counts for any entry it was provided, assuming a count of 1 for any entry it was not. A deeper analysis of organismQuantity and organismQuantityType could reveal better estimations of sighting numbers in future analysis.

As noted from data exploration, the counts for each year were heavily skewed as the years went on, especially for 2016. We decided to use a log transformation to standardize the counts and add more weight to years that had lower count totals. This later proved detrimental to our models and better performance was seen when this was removed.

Since we were trying to look at time series data between species, we needed to aggregate the numerical data from our dataset to help with predictions for the model. Using species name and year as the grouping criteria, we obtained average temperature, average salinity, average shore distance, min/max of latitude and longitude, average bathymetry, and the total counts of observation per year for each species. This information we would later use to compute more temporal data.

Throughout these preprocessing steps, we periodically had to remove null values to make sure our data was able to be used by the prediction models of our choosing. The most important columns for our analysis were environmental data related which was consistently provided by OBIS, but still had millions of missing entries. While some of it could be retrieved or assumed, as stated in the data exploration discussion section, some data was simply not provided. While accompanying data such as NOAA buoy sampling to incorporate rolling oceanographic chemical data, this proved to be beyond the scope of this project. So ultimately, by removing these null values, our dataset was heavily reduced in volume. 

#### IUCN Dataset

To incorporate the IUCN dataset it was necessary to first clean the provided csv file as the IUCN dataset was littered with comments and unformatted text columns. Rigorous manual cleaning was applied, with any further errors being dropped during the joining process.

With the new IUCN dataset handling null and outdated classifications, we wanted to create a binary column for easier classification. This was labeled as safe or unsafe (true or false which was later changed to 0 or 1). A species was labeled as safe if it was near threatened or least concerned and not safe otherwise. However we came to realize that leaving the classifications as the original 7 may have been better for the model. This fact was unfortunately never explored but might prove interesting as a follow up to this project.

#### Merged Dataset 

For our final pre-processing step we generated time series data. We included moving averages, lag 1 data, and year on year changes for multiple numerical variables. We kept our moving average to a range of 3 years due to limited scope of available data after removing anything older than the assessment years in our IUCN dataset. We did not look into lags more than 1 or multiple year or year changes which could have altered the accuracy of our model. We initially did not standardize any values besides sum counts, which contributed to an inaccurate performing first model. However, we included standardized values for later models.

### Model 1a - Random Forest

We chose a random forest because the ensemble method can better utilize multiple features that a decision tree would not be able to capture. Being able to better generalize certain relationships between these features would ideally outperform a singular tree model.  

Due to the random forest classification model specifications, we had to remove many columns that were not numerical or binary. This included many of the columns pertaining to the taxon of species such as kingdom, domain etc. These columns could have helped with the accuracy and prediction but we could not have tested this. It is possible that word embeddings may have been applied and should be considered for future applications. One-hot encoding would have caused the dimensionality to explode and would not be recommended. We also did not scale other numerical features which we later did for future models.

Looking at the validation errors the model appears to be doing extremely well. However, the model is very biased because the data has more safe labels over unsafe labels. The ratio of correctly predicted labels merely mirrored our data distribution showing that our model was not training correctly. In order to have a model that generalizes better we decided that we would need to reduce the number of safe species to fit 50% of the total data. In addition, we would reduce the number of features followed by a new grid-search and fitting graph which would provide a model that better generalizes to unseen data. 

### Model 1b - Random Forest v2

We decided to do another random forest model with the adjustments to the data (see discussion). With these adjustments, the model better generalized to the data but our accuracy suffered greatly. However this trade off was worth it as it could more accurately identify at-risk species as opposed to blanket classifying everything as safe. 

Additionally certain features started to become more relevant. Moving average sst (sea surface temperature) being a top feature could indicate that organisms in areas where climate change has a larger impact on sst (such as polar regions and tropical) might be facing the greatest threat. Moving average sum count is self explanatory as those numbers increasing and decreasing would have direct correlation on population counts. But the rolling standard deviation sea surface salinity being an impactful feature is hard to reason out. It could be that organisms with narrow salinity tolerances are more impacted by higher variability in their local salinity, similar to organisms being affected by changing sst, and this is reflected in the model. But this would require deeper analysis. Regardless, we wanted to try a different model that could potentially have a better accuracy given the same adjustments we made. 

### Model 2 - Gradient Boosted Trees

We chose gradient boosted trees as the second model as they could potentially handle outlying errors better than a random forest by iteratively correcting errors from previous weak learners. The random forest model struggled with many outliers, which we hoped gradient boosted trees would manage more effectively. Gradient boosted trees also tend to perform better on complex data, and given that our data is mostly environmental factors that have varied interactions over time, we expected gradient boosted trees to perform better.

The gradient boosted tree model showed similar accuracy to the model 1b. However, given that the run time for this model is much higher than random forest, it may not be worth the slight difference in accuracy. Additionally, the fitting graph showed our model had overfit in nearly all levels of depth, which might mean that the model would have difficulty in generalizing to new data. 

Despite this, some interesting results were gleaned from the feature importance extraction. Consistently, sst and sss (sea surface salinity) were among the top features in determining species red list category classification, very similar to the random forest models. Average shore distance being the most influential feature could be explained by human impact as organisms closer to shore might have more human interactions. Or it could be a bias in the data as those reporting this data are going to use observations they have the easiest access to. Bathymetry might have a similar story in that as a majority of marine species live in shallow waters to benefit from the phytoplankton requiring sunlight, they are also the easiest to be accessed by humans. This in turn could make them vulnerable to human activities, or it could make them easier to catalog, making them part of a skewed sampling process.

While the GBT classifier did not meet expectations, it did provide interesting insights into marine ecological relationships that would benefit from further analysis.

### Final Model - Random Forest v2

After considering the models and their similar performances, we decided to use the altered random forest model as our final model. Its fitting graph showed much more promise compared to the other models, it had a similar if not equal performance to the GBT model, as well as being able to be trained faster. These qualities made it the best candidate for our final model. While its test performance was not at the level we were hoping for, further feature expansion may improve performance in future iterations.

As seen in the prior models, sst and sss were consistently the most important features throughout the dataset. What was surprising was how spread counts were in their influence on the final model, with rolling averages being the highest impact. Year over year features (denoted yoy in the variable names), were consistently the least influential temporal features, which does make some sense as a species change from one year to the next is not representative of a larger trend. Additionally, location information (denoted lat and long in the variable names), was also a poorer predictor than expected. There was some anticipation that changing organism ranges might have a relationship with red list category classification. But that might have been a stretch to assume when such information might have been sourced from captive animal locations. As environmental factors seemed to play the largest role, using temporally dynamic environmental indicators such as dissolved oxygen, chlorophyll a, or nitrogen levels (estimated from satellite imagery) could provide further improvements to our models.
Conclusion
Models that focus on the ways in which marine biodiversity changes over time can help inform policy makers and scientists on how to make decisions about topics like fishery availability and conservation effort allocation. Expanding these models to data sets like GBIF (Global Biodiversity Information Facility) could push this idea beyond marine ecosystems, helping understand the ongoing effects of climate change and human impact on our environment.

However this predictive model is only a first step. An accuracy of 68% is not reliable enough to generalize to other populations. One limitation we faced was that our IUCN dataset only possessed the current classification of a given species. Expanding that to include older assessments of red list category classification might provide insights to long standing trends that could improve our model performance. Further data expansion could be done via merging oceanographic chemical data collected by NOAA to provide further indicators of a species changing status.

Our models also employed a binary classification. Classifying the full range of red list categories would be much more useful to an end user who might want a finer grain understanding of a species standing. Additionally, while we tested our model random forest and GBT classification, using more complex models such as neural networks or stacking various models may utilize hidden relationships that our current models might not be able to rely on. 

Finally, it is important to recognize that conflating global marine species environmental interactions to local fauna might not be an accurate reflection of the complex interactions happening within a select biome. Using this framework to transform local data on a case-by-case might provide a better understanding to how a species is reacting to changes in the environment over time. Careful consideration must be taken of how we can use machine learning models to help species conservation.

## Collaboration

Ekai: Co author and co coder.
- Contributed to both the code and write up.
-  Worked as a team. 
- Set up meetings
- 
Sanjana: Co author and co coder. 
- Contributed to both the code and write up. 
- Worked as a team. 
- Set up meetings
