# DSC232RGroupProject - OBIS Classification

<p align="center">
  <img src="https://github.com/sanrajagopal/DSC232RGroupProject/blob/8199477757f0599c336365e5d0e7a9e9daeb3a97/pics/sadturtle.png" alt="Kemp's ridley turtle washed up on the Isle of Harris Credit: Ruth A Hamilton">
</p>
<p align="center">
  <i>Figure 1. Kemp's ridley turtle washed up on the Isle of Harris. Credit: Ruth A Hamilton. Source: <a href="https://www.mcsuk.org/news/sad-stranded-turtles/" title="MCSUK" target="_blank">Link</a></i>
</p>

## Introduction

The OBIS (Ocean Biodiversity Information System) dataset is a collection of marine animal observations built from collected documentation, volunteer-provided observations, and aggregated data sets whose records go back over 100 years. This open-access data provides information on marine biodiversity to help highlight insights for scientific communities and generate potential sustainability practices. We chose this dataset because we were curious about what different attributes could affect a species' endangerment and whether we can predict if a species is or will be endangered with those features. The IUCN (International Union for Conservation of Nature) is an international organization that is primarily known for its red list categories. Below is an image showing the different red list categories: extinct, extinct in the wild, critically endangered, endangered, vulnerable, near threatened, and least concerned. These labels determine a species' risk worldwide of ceasing to exist. Our goal is to dive into the OBIS data set and create a classifier that can accurately predict whether a species may be at risk of becoming threatened according to the IUCN red list categories.

The OBIS data set can be downloaded here: [OBIS Data Set](https://obis.org/data/access/)

You can follow along in our Jupyter Notebook here: [Jupyter Notebook](https://github.com/sanrajagopal/DSC232RGroupProject/blob/main/obis.ipynb)

<p align="center">
  <img src="https://github.com/sanrajagopal/DSC232RGroupProject/blob/3b2ca8e3a25818388ae9325fc618afb05c05536f/pics/iucnclass.png" alt="Red list category rankings">
</p>
<p align="center">
  <i>Figure 2. Red list category rankings according to the IUCN (EX: Extinct, EW: Extinct in the Wild, CE: Critically Endangered, EN: Endangered, VU: Vulnerable, NT: Near Threatened, LC: Least Concern). Source: <a href="https://commons.wikimedia.org/w/index.php?curid=1493206" title="Wikipedia" target="_blank">Link</a></i>
</p>

## Method Section

### Data Exploration

The size of our dataset was a 17GB parquet file after compression, necessitating the use of Pyspark to perform much of the data exploration. A minimum of 8gb per core was required. The project was completed in a jupyter notebook run on a cluster provided by the SDSC (San Diego Supercomputer Center). The following libraries were used throughout the project.
```python
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
```

We first looked into the number of observations per year for the dataset by grouping by the year column and aggregating the counts. We then filtered the dates after 1899 to get all dates from the 20th to 21st centuries. This was then convereted into a pandas dataframe which was used to create a histogram to display the results (see Results - Figure 3). Below is a compressed version of the code used.
```python
df_dates = df.groupBy(df.date_year).agg({'date_year':'count'}).sort(df.date_year).cache()
df_dates = df_dates.withColumnRenamed('count(date_year)','obs_per_year')
pandas_df_dates = df_dates.toPandas()
pandas_df_dates_cut = pandas_df_dates[pandas_df_dates.date_year > 1899]
plt.bar(pandas_df_dates_cut.date_year,pandas_df_dates_cut.obs_per_year)
plt.title("Number of Observations per Year")
plt.xlabel("Years")
plt.ylabel("Observations(per 10 million)")
```

Then we looked into how many red list categories per species there were in the whole dataset. Filtering for unique species and red list category pairs, followed by grouping by the red list category column alone found us the counts per category. This summarized table was then converted into a pandas dataframe. We then used the dataframe to create a bar graph to display the counts of the red list categories (see Results - Figure 5). Below is a compressed version of the code used.
```python
df_redlist_distinct = df.select("scientificName","redlist_category").filter(df.scientificName != "").filter(df.redlist_category != "").distinct().cache()
df_redlist_count = df_redlist_distinct.select("redlist_category").groupBy("redlist_category").agg({'redlist_category':'count'})
df_redlist_count = df_redlist_count.withColumnRenamed('count(redlist_category)','obs_per_cat')
pandas_redlist_count = df_redlist_count.toPandas()
plt.bar(pandas_redlist_count.redlist_category,pandas_redlist_count.obs_per_cat)
plt.title("Number of Distinct Species per IUCN classification")
plt.xlabel("IUCN Classification")
plt.ylabel("Number of species")
```

To compare trends, we decided to look into a well-known marine species, Orinicus Orcas (Orcas aka Killer Whales), and track the number of observations made per year. We filtered by species name matching Ornicus Orcas, grouped by year, and aggregated the counts, and converted the results to a pandas dataframe like before. The line graph generated showed how observations changed over time (see results Figure 4). Below is a compressed version of the code used.
```python
df_orca = df.filter(df.scientificName == "Orcinus orca").select("date_year").groupBy("date_year").agg({'date_year':'count'}).sort("date_year").cache()
df_orca = df_orca.withColumnRenamed('count(date_year)','obs_per_year')
pandas_orca = df_orca.toPandas()
plt.plot(pandas_orca.date_year,pandas_orca.obs_per_year)
plt.title("Orca Observations over Time")
plt.xlabel("Year")
plt.ylabel("Number of observations")
```

To better prepare for data preprocessing we checked to see how many null values were in the dataset. We created a new data frame that took each column and summed the value of nulls per column. Below is a compressed version of the code used.
```python
null_counts = df.select([f.sum(f.col(column).isNull().cast("int")).alias(column) for column in df.columns])
null_counts_dict = null_counts.collect()[0].asDict()
null_counts_df = sc.createDataFrame(null_counts_dict.items(), ["column_name", "null_count"])
num_rows = null_counts_df.count()
null_counts_df.show(num_rows, truncate=False)
```

Here is a sample of feature descriptions as provided by OBIS ([Link](https://obis.org/data/access/)). There were some features not mentioned in this original list that were used in later analysis such as ‘eventDate’. In total, there were 268 columns.
- id: Globally unique identifier assigned by OBIS.
- dataset_id: Internal dataset identifier assigned by OBIS.
- decimalLongitude: Parsed and validated by OBIS.
- decimalLatitude: Parsed and validated by OBIS.
- date_start: Unix timestamp based on eventDate (start).
- date_mid: Unix timestamp based on eventDate (middle).
- date_end: Unix timestamp based on eventDate (end).
- date_year: Year based on eventDate.
- scientificName: Valid scientific name based on the scientificNameID or derived by matching the provided scientificName with WoRMS
- originalScientificName: The scientificName as provided.
- minimumDepthInMeters: Parsed and validated by OBIS.
- maximumDepthInMeters: Parsed and validated by OBIS.
- coordinateUncertaintyInMeters: Parsed and validated by OBIS.
- flags: Quality flags added by OBIS. The quality flags are documented here.
- dropped: Record dropped by OBIS quality control?
- absence: Absence record?
- shoredistance: Distance from shore in meters added by OBIS quality control, based on OpenStreetMap. Negative value indicates that the observation was inland by -1 times that distance
- bathymetry: Bathymetry added by OBIS. Bathymetry values based on EMODnet Bathymetry and GEBCO, see https://github.com/iobis/xylookup (Data references)
- sst: Sea surface temperature added by OBIS. sst values based on Bio-Oracle, see https://github.com/iobis/xylookup (Data references)
- sss: Sea surface salinity added by OBIS. sss values based on Bio-Oracle, see https://github.com/iobis/xylookup (Data references)
- marine: Marine environment flag based on WoRMS.
- brackish: Brackish environment flag based on WoRMS.
- freshwater: Freshwater environment flag based on WoRMS.
- terrestrial: Terrestrial environment flag based on WoRMS.
- taxonRank: Based on WoRMS.
- AphiaID: AphiaID for the valid name based on the scientificNameID or derived by matching the provided scientificName with WoRMS.
- redlist_category: IUCN Red List category.

### Preprocessing

#### OBIS Dataset

As stated in the data exploration step, due to the volume of data a series of cleaning operations was required.

**Remove irrelevant columns** - Numerous columns showed duplicate information, did not have a variable description provided/could not be deciphered, or did not provide useful insight, so a majority of these columns were removed. Below is the code used.
```python
df_trim_col = df.select('id','decimalLongitude','decimalLatitude','date_year','scientificName','coordinateUncertaintyInMeters','shoredistance','bathymetry','sst','sss','marine','brackish','freshwater','terrestrial','taxonRank','redlist_category','superdomain','domain','kingdom','subkingdom','infrakingdom','phylum','phylum_division','subphylum_subdivision','subphylum','infraphylum','parvphylum','gigaclass','megaclass','superclass','class','subclass','infraclass','subterclass','superorder','order','suborder','infraorder','parvorder','superfamily','family','subfamily','supertribe','tribe','subtribe','genus','subgenus','section','subsection','series','species','subspecies','natio','variety','subvariety','forma','subforma','individualCount','eventDate')
```

**Extracted year from dates** - We wanted to group the entries by the species name as well as the year of the sighting. However, some entries were missing available year information which we extracted from a user-entered date column and inserted into the available year column. A UDF was required to parse the dates which proved to be slow. Below is the code used.
```python
# UDF
from dateutil import parser
def extract_year(date_str):
    try:
        parsed_date = parser.parse(date_str, fuzzy=True)
        return parsed_date.year
    except:
        return None
extract_year_udf = f.udf(extract_year, IntegerType())

# Year Extraction
df_year_fix = df_trim_col.withColumn(
    "date_year",
    f.when(f.col("date_year").isNull(), extract_year_udf(f.col("eventDate")))
    .otherwise(f.col("date_year"))
).drop("eventDate").filter(f.col("date_year").isNotNull())
```

**Handling nulls and uncertainty** - Null values in the data set were dropped. If data with outdated naming was found, it was replaced with the current scientific name. In situations in which the data showed signs of excessive uncertainty, attributed to the coordinateUncertaintyInMeters feature, the data was dropped as well. Below is the code used.
```python
df_sci_name_fix = df_year_fix.filter(f.col("scientificName").isNotNull()).withColumn("scientificName", f.regexp_replace('scientificName', 'Taenioides jacksoni', 'Trypauchenopsis intermedia'))
df_uncert_fix = df_sci_name_fix.filter((f.col("coordinateUncertaintyInMeters") <= 1000) | (f.col("coordinateUncertaintyInMeters").isNull()))
df_uncert_fix = df_uncert_fix.withColumn(
    "coordinateUncertaintyInMeters",
    f.when(f.col("coordinateUncertaintyInMeters").isNull(), f.col("coordinateUncertaintyInMeters") == 0)).drop("coordinateUncertaintyInMeters")
df_env_fix = df_uncert_fix.filter(f.col("sst").isNotNull()).filter(f.col("sss").isNotNull()).filter(f.col("bathymetry").isNotNull())
```

**Managing counts** - Values in the individualCount feature were converted into an integer and used to adjust a count column if the data was present, using a count of 1 otherwise. Below is the code used.
```python
df_indiv_cnt_fix = df_env_fix.withColumn('individualCount', 
                                         f.when(f.col('individualCount').isNotNull(), 
                                         f.col('individualCount').cast('int'))
                                         .otherwise(None)
                                        )
df_count_fix = df_indiv_cnt_fix.withColumn(
    'count',
    f.when((f.col('individualCount').isNull()) | (f.col('individualCount') <= 1), 1)
    .when((f.col('individualCount') > 1), f.col('individualCount'))
    .otherwise(None)
).drop('individualCount')
```

**Aggregation columns** - Much of the numerical data had to be aggregated by year and species so multiple new columns were created such as average temperature, average salinity, average shore distance, min/max of latitude and longitude, average bathymetry, and the total counts of observation per year for each species. Below is the code used.
```python
aggregated_df = df_count_fix.groupBy("scientificName", "date_year").agg(
    f.min("decimalLongitude").alias("min_decimalLongitude"),
    f.max("decimalLongitude").alias("max_decimalLongitude"),
    f.min("decimalLatitude").alias("min_decimalLatitude"),
    f.max("decimalLatitude").alias("max_decimalLatitude"),
    f.avg("shoredistance").alias("avg_shoredistance"),
    f.avg("bathymetry").alias("avg_bathymetry"),
    f.avg("sst").alias("avg_sst"),
    f.avg("sss").alias("avg_sss"),
    f.sum("count").alias("sum_count")
)
```

**Scale to observation counts** - The number of observations varied, especially comparing older years to recent ones so we introduced a log transformation on counts. This was later removed as a scaler was applied to all data in future models. Below is the code used.
```python
aggregated_df_log_trans= aggregated_df.withColumn("log_sum_count", f.log1p("sum_count"))
```

**Merge with lost information** - The cladogram and environmental territory information lost as part of the aggregation process was joined back in. Below is the code used.
```python
df_clado_unique = df_year_fix.select('date_year','scientificName','marine','brackish','freshwater','terrestrial','superdomain','domain','kingdom','subkingdom','infrakingdom','phylum','phylum_division','subphylum_subdivision','subphylum','infraphylum','parvphylum','gigaclass','megaclass','superclass','class','subclass','infraclass','subterclass','superorder','order','suborder','infraorder','parvorder','superfamily','family','subfamily','supertribe','tribe','subtribe','genus','subgenus','section','subsection','series','species','subspecies','natio','variety','subvariety','forma','subforma',).dropDuplicates(['scientificName', 'date_year'])
df_merge_clean = aggregated_df_log_trans.join(df_clado_unique, on=["scientificName", "date_year"], how="left")
```

#### IUCN Dataset

To have accurate IUCN red list categories for the species we incorporated the IUCN red list assessment dataset which has up-to-date red list assessments. Each species the IUCN has assessed is given a classification and its latest assessment date. This information would act as the labels for our data.

**Extracted year from dates** - Similar to the previous dataset, we extracted only the year using the UDF. Below is the code used.
```python
df_assessments = df_iucn.withColumn("assessmentYear",extract_year_udf(f.col("assessmentDate")))
df_assessments = df_assessments.drop('assessmentDate')
```

**Create the safe column** - For the labels, we created a boolean column that would determine whether a species is considered safe or unsafe. If the species was extinct, extinct in the wild, critically endangered, endangered, or vulnerable it would be labeled as not safe (false). If the species was near threatened or least concerned the label would be safe (true). This was later changed to 0 for false and 1 for true so that our models could classify the labels appropriately. Below is the code used.
```python
df_assessments = df_assessments.filter(f.col('redlistCategory').isNotNull())
df_assessments = df_assessments.filter(f.col('redlistCategory') != "Data Deficient")
df_assessments = df_assessments.withColumn("safe?",f.when((df_assessments.redlistCategory == "Endangered") | (df_assessments.redlistCategory == "Extinct")| (df_assessments.redlistCategory == "Vulnerable") | (df_assessments.redlistCategory == "Critically Endangered") |(df_assessments.redlistCategory == "Extinct in the wild"),False).otherwise(True))
df_assessments = df_assessments.drop("redlistCategory")
```

#### Merged Dataset

We merged the preprocessed OBIS and IUCN datasets to create a new dataset that would be used for our models. This was followed by some additional cleaning.

**Dropped species with NULL safe** - If a species did not have a red list category at this point, they were removed. Below is the code used.
```python
df_merge_safe = df_merge_asses.filter(f.col("safe?").isNotNull())
```

**Feature expansion** - Using the aggregation columns that were created, we created new features that showed the year-on-year changes of aggregates, moving averages, and rolling standard deviations. Below is the code used.
```python
windowSize = 2
windowSpec = Window.partitionBy("scientificName").orderBy("date_year").rowsBetween(-windowSize, 0)
lagWindowSpec = Window.partitionBy("scientificName").orderBy("date_year")
df_merge_years = df_merge_safe.withColumn("moving_avg_sst", f.avg("avg_sst").over(windowSpec))
df_merge_years = df_merge_years.withColumn("rolling_stddev_sst", f.stddev("avg_sst").over(windowSpec))
df_merge_years = df_merge_years.withColumn("lag1_avg_sst", f.lag("avg_sst", 1).over(lagWindowSpec))
df_merge_years = df_merge_years.withColumn("yoy_change_avg_sst", f.col("avg_sst") - f.col("lag1_avg_sst"))
df_merge_years = df_merge_years.drop("lag1_avg_sst")
df_merge_years = df_merge_years.withColumn("moving_avg_sss", f.avg("avg_sss").over(windowSpec))
df_merge_years = df_merge_years.withColumn("rolling_stddev_sss", f.stddev("avg_sss").over(windowSpec))
df_merge_years = df_merge_years.withColumn("lag1_avg_sss", f.lag("avg_sss", 1).over(lagWindowSpec))
df_merge_years = df_merge_years.withColumn("yoy_change_avg_sss", f.col("avg_sss") - f.col("lag1_avg_sss"))
df_merge_years = df_merge_years.drop("lag1_avg_sss")
df_merge_years = df_merge_years.withColumn("moving_avg_sum_cnt", f.avg("sum_count").over(windowSpec))
df_merge_years = df_merge_years.withColumn("rolling_stddev_sum_cnt", f.stddev("sum_count").over(windowSpec))
df_merge_years = df_merge_years.withColumn("lag1_sum_cnt", f.lag("sum_count", 1).over(lagWindowSpec))
df_merge_years = df_merge_years.withColumn("yoy_change_sum_cnt", f.col("sum_count") - f.col("lag1_sum_cnt"))
df_merge_years = df_merge_years.drop("lag1_sum_cnt")
df_merge_years = df_merge_years.withColumn("moving_avg_log_cnt", f.avg("log_sum_count").over(windowSpec))
df_merge_years = df_merge_years.withColumn("rolling_stddev_log_cnt", f.stddev("log_sum_count").over(windowSpec))
df_merge_years = df_merge_years.withColumn("lag1_log_cnt", f.lag("log_sum_count", 1).over(lagWindowSpec))
df_merge_years = df_merge_years.withColumn("yoy_change_log_cnt", f.col("log_sum_count") - f.col("lag1_log_cnt"))
df_merge_years = df_merge_years.drop("lag1_log_cnt")
temp = df_merge_years.withColumn("difference_long", f.col("max_decimalLongitude")-f.col("min_decimalLongitude"))
temp1 = temp.withColumn("difference_lat", f.col("max_decimalLatitude")-f.col("min_decimalLatitude"))
temp2 = temp1.withColumn("yoy_dif_lat", f.col('difference_lat')-f.lag(f.col('difference_lat')).over(lagWindowSpec))
temp3 = temp2.withColumn("yoy_dif_long", f.col('difference_long')-f.lag(f.col('difference_long')).over(lagWindowSpec))
temp4 = temp3.withColumn("yoy_dif_shoredistance", f.col('avg_shoredistance')-f.lag(f.col('avg_shoredistance')).over(lagWindowSpec))
df_temporal = temp4.withColumn("yoy_dif_bath", f.col('avg_bathymetry')-f.lag(f.col('avg_bathymetry')).over(lagWindowSpec))
df_temporal = df_temporal.drop(
    'min_decimalLongitude',
    'max_decimalLongitude',
    'min_decimalLatitude',
    'max_decimalLatitude',
)
```

**Filtering years before assessment year** - Since we are not able to determine the accuracy of a red list category before the year of assessment, those rows were removed. Below is the code used.
```python
df_temporal = df_temporal.filter(f.col("date_year") >= f.col("assessmentYear")).sort("scientificName", "date_year").drop("assessmentYear").cache()
```

### Model 1a - Random Forest

#### Model Creation 

The first model we chose was a random forest model. Due to the necessity for the random forest model to ingest numerical data only a number of columns had to be editted or removed. We set up a vector assembler that took in the features that were specified from the final processed dataset. Then the data was split up to train/validation/test with a 0.6/0.2/0.2 split. Using our training and validation data, we performed a grid search on the hyperparameters max depth and num trees. This code was then re-adapted to form the code for the fitting graph. Using max depth as our basis for the fitting graph, we then checked the fit of our model (see results Figure 6).

Here is a compressed version of the code used to transform the data for random forest.
```python
df_temporal_rf = df_temporal.withColumn("marine", f.col("marine").cast("int")).withColumn("brackish", f.col("brackish").cast("int")).withColumn("terrestrial", f.col("terrestrial").cast("int")).withColumn("freshwater", f.col("freshwater").cast("int")).withColumn("safe?", f.col("safe?").cast("int")).fillna(0)
features = ['avg_shoredistance','avg_bathymetry','avg_sst','avg_sss','sum_count','log_sum_count','marine','brackish','freshwater','terrestrial','moving_avg_sst','rolling_stddev_sst','yoy_change_avg_sst','moving_avg_sss','rolling_stddev_sss','yoy_change_avg_sss','moving_avg_sum_cnt','rolling_stddev_sum_cnt','yoy_change_sum_cnt','moving_avg_log_cnt','rolling_stddev_log_cnt','yoy_change_log_cnt','difference_long','difference_lat','yoy_dif_lat','yoy_dif_long','yoy_dif_shoredistance','yoy_dif_bath']
assembler = VectorAssembler(inputCols=features, outputCol="features")
df_ml_vector = assembler.transform(df_temporal_rf).select('safe?', 'features')
train_data, validation_data, test_data = df_ml_vector.randomSplit([0.6, 0.2, 0.2], seed=42)
train_data = train_data.cache()
validation_data = validation_data.cache()
test_data = test_data.cache()
```

Here is a compressed version of the code used to check the error rate at multiple depths.
```python
maxDepth_list = [3, 5, 10, 15, 20, 30]
evaluator = MulticlassClassificationEvaluator(labelCol="safe?", predictionCol="prediction", metricName="accuracy")
results = []
for depth in maxDepth_list:
    rf = RandomForestClassifier(featuresCol="features", labelCol="safe?", maxDepth=depth,seed = 42)
    model = rf.fit(train_data)
    train_predictions = model.transform(train_data)
    train_accuracy = evaluator.evaluate(train_predictions)
    val_predictions = model.transform(validation_data)
    val_accuracy = evaluator.evaluate(val_predictions)
    results.append({'parameters': f"Depth {depth}",'train_error': 1 - train_accuracy,'val_error': 1 - val_accuracy})
```

Here is a compressed version of the code used to plot the fitting graph.
```python
parameters = [r['parameters'] for r in results]
train_errors = [r['train_error'] for r in results]
val_errors = [r['val_error'] for r in results]
plt.figure(figsize=(12, 6))
plt.plot(parameters, train_errors, label='Training Error', marker='o')
plt.plot(parameters, val_errors, label='Validation Error', marker='o')
plt.xlabel('Model Parameters (Max Depth)')
plt.ylabel('Error Rate')
plt.title('Fitting Graph of Random Forest Models')
plt.xticks(rotation=45, ha = 'right')
plt.legend()
plt.tight_layout()
plt.show()
```

#### Ground Truth

Once the parameters were chosen, the selected model was trained again on the training data set, and the final validation error was determined (see results). To verify these results we looked at a ground truth example. We first selected a safe observation and an unsafe observation. From that, we created a new data frame that showed whether it came from the training set or the validation  set, the features, the actual values, the predicted values, and a new column that showed whether the predicted values were correct or not. 

Here is a compressed version of the code used to obtain the training and validation errors of the select model.
```python
rf_10 = RandomForestClassifier(featuresCol="features",labelCol="safe?",maxDepth=10,seed = 42)
model_10 = rf_10.fit(train_data)
evaluator = MulticlassClassificationEvaluator(labelCol="safe?", predictionCol="prediction", metricName="accuracy")
train_predictions = model_10.transform(train_data).withColumn("dataset", f.lit("Train"))
train_error = 1 - evaluator.evaluate(train_predictions)
validation_predictions = model_10.transform(validation_data).withColumn("dataset", f.lit("Validation"))
validation_error = 1 - evaluator.evaluate(validation_predictions)
```

Here is a compressed version of the code used to look at a ground truth examples.
```python
train_pred_safe = train_predictions.select('dataset','features', 'safe?', 'prediction').filter(f.col('safe?') == 1).first()
train_pred_unsafe = train_predictions.select('dataset','features', 'safe?', 'prediction').filter(f.col('safe?') == 0).first()
val_pred_safe = validation_predictions.select('dataset','features', 'safe?', 'prediction').filter(f.col('safe?') == 1).first()
val_pred_unsafe = validation_predictions.select('dataset','features', 'safe?', 'prediction').filter(f.col('safe?') == 0).first()
ground_df = sc.createDataFrame([train_pred_safe, train_pred_unsafe,val_pred_safe,val_pred_unsafe])
ground_df.withColumn("correct", f.when(f.col('safe?') == f.col('prediction'), True).otherwise(False)).show()
```

#### Confusion Matrix 

A confusion matrix was created by selecting the actual labels vs predicted values from the validation set. Because data frames in Pyspark do not have a direct confusion matrix function, we utilized the function from pyspark.mllib instead. This required the data to be transformed into an RDD at which point it could be made into a confusion matrix and presented using seaborn (see results Figure 7).

Here is a compressed version of the code used to create the confusion matrix.
```python
predictionAndLabels = validation_predictions.select(f.col("prediction"), f.col("safe?"))
predictionAndLabelsRDD = predictionAndLabels.rdd.map(lambda x: (float(x[0]), float(x[1])))
metrics = MulticlassMetrics(predictionAndLabelsRDD)
confusionMatrix = metrics.confusionMatrix().toArray()
confusion_df = pd.DataFrame(confusionMatrix, index=['Actual 0', 'Actual 1'], columns=['Predicted 0', 'Predicted 1'])
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_df, annot=True, cmap="viridis")
plt.title("Random Forest Confusion Matrix Heatmap")
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.show()
```

#### Feature Importance 

To determine the impact of each feature of the model, we extracted the provided feature importances using the available model method. Ranking and graphing the importance of each feature via a histogram displayed the strength of our selections (see results Figure 8).

Here is a compressed version of the code used to create the feature histogram.
```python
importances = model_10.featureImportances
importance_list = importances.toArray()
feature_importance_pairs = list(zip(features, importance_list))
feature_importance_pairs.sort(key=lambda x: x[1], reverse=True)
sorted_features, sorted_importance_list = zip(*feature_importance_pairs)
plt.figure(figsize=(10, 6))
plt.bar(sorted_features, sorted_importance_list)
plt.xlabel('Features')
plt.ylabel('Importance')
plt.title('Random Forest Feature Importances')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()
```

### Model 1b - Random Forest v2

From the issues that were discussed in the discussion section (see discussion), we decided to do another Random Forest model with adjustments to combat these issues. 

We first reduced the amount of safe values by sampling 0.0888 of the safe values from the dataset. This resulted in a 50/50 split of safe versus non-safe labels. Then we removed the least important features which were marine, freshwater, terrestrial, and brackish. We also removed all features involving log transformations. Then we scaled all the features using the MinMaxScaler function. Then using the same train/validation/test split from before, a parameter grid of max depth and number of trees, alongside a fitting graph built off of max depth (see results Figure 10), we found the best model that was then used to create the final model. A ground truth table, a confusion matrix (see results Figure 10) and an importance feature graph (see results Figure 11) were also created with the same methods discussed in model 1a. 

Here is a compressed version of the code used to further transform the data for Model 1b. Please refer to Method Section - Model 1a or the [Jupyter Notebook](https://github.com/sanrajagopal/DSC232RGroupProject/blob/main/obis.ipynb) for the code for the fitting graph, ground truth, confusion matrix, and feature importance.
```python
df_temporal_notsafe_rf = df_temporal.filter(f.col('safe?')==0)
df_temporal_safe_rf = df_temporal.filter(f.col('safe?')==1)
df_temporal_safe_rf = df_temporal_safe_rf.sample(False, fraction=0.0888, seed=42)
df_temporal_rfv2 = df_temporal_safe_rf.union(df_temporal_notsafe_rf).withColumn("safe?", f.col("safe?").cast("int")).fillna(0)
features = ['avg_shoredistance','avg_bathymetry','avg_sst','avg_sss','sum_count','moving_avg_sst','rolling_stddev_sst','yoy_change_avg_sst','moving_avg_sss','rolling_stddev_sss','yoy_change_avg_sss','moving_avg_sum_cnt','rolling_stddev_sum_cnt','yoy_change_sum_cnt','difference_long','difference_lat','yoy_dif_lat','yoy_dif_long','yoy_dif_shoredistance','yoy_dif_bath']
assembler = VectorAssembler(inputCols=features, outputCol="features")
df_rfv2_vector = assembler.transform(df_temporal_rfv2)
scaler = MinMaxScaler(inputCol='features', outputCol='scaled_features')
scaler_model_rf = scaler.fit(df_rfv2_vector)
df_scaled_rfv2 = scaler_model_rf.transform(df_rfv2_vector)
train_data_rf2, validation_data_rf2, test_data_rf2 = df_scaled_rfv2.select('safe?', 'scaled_features').randomSplit([0.6, 0.2, 0.2], seed=42)
train_data_rf2 = train_data_rf2.cache()
validation_data_rf2 = validation_data_rf2.cache()
test_data_rf2 = test_data_rf2.cache()
```

### Model 2 - Gradient Boosted Trees Classifier

Using the same adjustments to the data made for our 2nd version of the random forest model, we decided to use a gradient-boosted tree classifier. While many of the same methods were employed here as was used in model 1b, the parameter grid was expanded to accommodate the GBT classifier's larger hyperparameter selection. Focusing on max depth, number of iterations, and step size for our parameter grid, and max depth for our fitting graph (see results Figure 12) we narrowed down to a model that was used in our final analysis. Like before, a ground truth table, a confusion matrix (see results Figure 13) and importance feature graphs (see results Figure 14) were created with the same methods discussed in model 1a.

Here is a compressed version of the code used to check the error rate at multiple depths of a GBT Classifier. Please refer to Method Section - Model 1b or the [Jupyter Notebook](https://github.com/sanrajagopal/DSC232RGroupProject/blob/main/obis.ipynb) for the code for the data augmentation process, plotting the fitting graph, ground truth, confusion matrix, and feature importance.
```python
maxDepth_list = [3, 5, 10, 15, 20, 30]
evaluator = MulticlassClassificationEvaluator(labelCol='safe?', predictionCol='prediction', metricName='accuracy')
gbt_results = []
for depth in maxDepth_list:
    gbt = GBTClassifier(labelCol='safe?', featuresCol='scaled_features',maxDepth=depth,maxIter=150,stepSize=0.1,seed=42)
    model_gbt = gbt.fit(train_data_gbt)
    train_predictions = model_gbt.transform(train_data_gbt)
    train_accuracy = evaluator.evaluate(train_predictions)
    val_predictions = model_gbt.transform(validation_data_gbt)
    val_accuracy = evaluator.evaluate(val_predictions)
    gbt_results.append({'parameters': f"Depth {depth}",'train_error': 1 - train_accuracy,'val_error': 1 - val_accuracy})
```

### Final Model - Random Forest v2

After reviewing the various models, we stuck with our random forest model using the altered data set as our final model.

As before, we retained the 50/50 split between safe and unsafe values to remove errors associated with distribution. This was paired with the removal of the least impactful features according to the feature importance metric of model 1a, which were marine, freshwater, terrestrial, and brackish. Finally, since we applied a min-max scaler to our data, we removed all count features that were put through a log transformation. This left us with the following 20 features: 'avg_shoredistance', 'avg_bathymetry', 'avg_sst', 'avg_sss', 'sum_count', 'moving_avg_sst', 'rolling_stddev_sst', 'yoy_change_avg_sst', 'moving_avg_sss', 'rolling_stddev_sss', 'yoy_change_avg_sss', 'moving_avg_sum_cnt', 'rolling_stddev_sum_cnt', 'yoy_change_sum_cnt', 'difference_long', 'difference_lat', 'yoy_dif_lat', 'yoy_dif_long', 'yoy_dif_shoredistance', 'yoy_dif_bath'.  However, unlike the previous models, we trained the model on the combination of training and validation data and tested our models on the test data set.

We checked the same model metrics as before looking at raw error performance, ground truth error, a confusion matrix (see results Figure 15), and ranked feature importance (see results Figure 16). These were obtained in the same method as previous models. However, a fitting graph was not checked in this instance as we were looking at the final model and its selected parameters rather than a range of parameters and their effect on fit. Please refer to Method Section - Model 1b or the [Jupyter Notebook](https://github.com/sanrajagopal/DSC232RGroupProject/blob/main/obis.ipynb) for the code.

## Results Section

### Data Exploration

From our preliminary data exploration, we found that the number of entries has increased over the years, with a noticeable spike occurring in 2016 with a diminishing count around 2019 onward. A similar trend could be observed when looking at a singular species, Orcinus orca (Orcas aka Killer Whales).

These counts do not reflect actual species counts as some entries represent groups of species observed as opposed to singular observations.

<p align="center">
  <img src="https://github.com/sanrajagopal/DSC232RGroupProject/blob/af4b4c7ef8eead00d05631c9e60773087de6ff32/pics/resultexp1.png" alt="Number of observations made per year starting at 1900">
</p>
<p align="center">
  <i>Figure 3. Number of observations (10 million) made per year starting at 1900. </i>
</p>

<p align="center">
  <img src="https://github.com/sanrajagopal/DSC232RGroupProject/blob/af4b4c7ef8eead00d05631c9e60773087de6ff32/pics/resultexp2.png" alt="Number of orca observations made per year">
</p>
<p align="center">
  <i>Figure 4. Number of orca observations made per year. </i>
</p>

As our goal was to use this data set to predict and classify at-risk species, we first summarized the total unique species with red list categories, finding 1979 instances. We found that the majority of the species were classified as NT (Not Threatened) and VU (Vulnerable). This also showed us that the OBIS dataset possessed outdated classifications which would need to be addressed during data pre-processing. This information combined with the red list category order, informed us of our cutoff between a “safe” and “unsafe” species to be NT (Not Threatened). NT (Not Threatened) and LC (Least Concern) acted as our “safe” categories, while everything else acted as our “unsafe” category. LC (Least Concern) is not on the graph below but its information was added later.

<p align="center">
  <img src="https://github.com/sanrajagopal/DSC232RGroupProject/blob/af4b4c7ef8eead00d05631c9e60773087de6ff32/pics/resultexp3.png" alt="Species per IUCN red list category classification">
</p>
<p align="center">
  <i>Figure 5. Species per IUCN red list category classification. LR/nt (Lower Risk/not threatened) and LR/cd (Lower Risk/conservation dependent) are outdated classifications used in older data or species who have not been reclassified since the change.</i>
</p>

Finally, we examined the null counts. With nearly 120 million entries (119568135) and 268 columns, null values were very common. The largest number of null values came from taxonomic entries, with the fewest being environmental data added in by OBIS (see discussion).

### Preprocessing

Before handling null values, a total of 210 columns were removed. From that point, years retrieved from user input columns helped retain 500,000 entries, but after careful removal of null values, we were left with a little over 100 million entries. After obtaining various aggregates grouped by species and year, we were left with 1.4 million entries.

While the OBIS dataset did provide some red list category classifications (approximately 2000 unique classified species), using the cleaned, full IUCN dataset expanded that to 150,000 possible unique classified species. Further feature expansion of the newly aggregated dataset added 18 columns containing previously inaccessible temporal features (3-year window). With the removal of any entries whose classification cannot be verified due to being older than the classification assessment year left us with 26859 entries and 28 features.

### Model 1a - Random Forest

After hyperparameter tuning via gridsearch on max depth and number of trees we found that a depth of 10 and the default tree count of 20 provided the best accuracy. Our fitting graph coincided with this showing a good balance between training and validation errors around this range.

<p align="center">
  <img src="https://github.com/sanrajagopal/DSC232RGroupProject/blob/af4b4c7ef8eead00d05631c9e60773087de6ff32/pics/model1a1.png" alt="Model 1a - Random Forest Fitting Graph">
</p>
<p align="center">
  <i>Figure 6. Model 1a - Random Forest Fitting Graph. Comparing the prediction error rate of the random forest model between the training and validation data sets over increasingly complex models (increasing maxDepth hyperparameter values) to check for model fit.</i>
</p>

This initial random forest model which had been trained on the fully cleaned dataset produced the following errors:

Train Error: 0.07325153374233129

Validation Error: 0.08239489489489493

Ground testing and a confusion matrix showed very few accurate predictions of “unsafe” species (safe?=0), but highly accurate ”safe” species (safe?=1)

<p align="center">
  <img src="https://github.com/sanrajagopal/DSC232RGroupProject/blob/af4b4c7ef8eead00d05631c9e60773087de6ff32/pics/model1a2.png" alt="Model 1a - Random Forest Confusion Matrix">
</p>
<p align="center">
  <i>Figure 7. Model 1a - Random Forest Confusion Matrix. Comparing actual labels and prediction labels for the validation data set.</i>
</p>

The top 3 features based on importance were: avg_shoredistance (0.07029372108800737), avg_sst (0.054585481271653045), yoy_dif_bath (0.053229316759500746)

The bottom 3 features based on importance were: brackish (0.005732192189060649), freshwater (0.00421483814423637), marine (0.0031321869659942663)

<p align="center">
  <img src="https://github.com/sanrajagopal/DSC232RGroupProject/blob/af4b4c7ef8eead00d05631c9e60773087de6ff32/pics/model1a3.png" alt="Model 1a - Random Forest Importance Features">
</p>
<p align="center">
  <i>Figure 8. Model 1a - Random Forest Importance Features. Importance features extracted from the model ranked in descending order.</i>
</p>

### Model 1b - Random Forest v2

After making the adjustments to the data, hyperparameter tuning via grid search found a depth of 10 and the default tree count of 20 to work best. Our fitting graph showed something similar, but it suggested a depth of 5 might better generalize to data.

<p align="center">
  <img src="https://github.com/sanrajagopal/DSC232RGroupProject/blob/af4b4c7ef8eead00d05631c9e60773087de6ff32/pics/model1b1.png" alt="Model 1b - Random Forest v2 Fitting Graph">
</p>
<p align="center">
  <i>Figure 9. Model 1b - Random Forest v2 Fitting Graph. Comparing the prediction error rate of the random forest model between the training and validation data sets over increasingly complex models (increasing maxDepth hyperparameter values) to check for model fit.</i>
</p>

Sticking with a depth of 10, and training the random forest model on a better-distributed data set with scaling applied produced the following errors:
Train Error: 0.08450704225352113

Validation Error: 0.31431767337807603

Ground testing and a confusion matrix showed semi-accurate predictions of “unsafe” species (safe?=0) and ”safe” species (safe?=1)

<p align="center">
  <img src="https://github.com/sanrajagopal/DSC232RGroupProject/blob/af4b4c7ef8eead00d05631c9e60773087de6ff32/pics/model1b2.png" alt="Model 1b - Random Forest v2 Confusion Matrix">
</p>
<p align="center">
  <i>Figure 10. Model 1b - Random Forest v2 Confusion Matrix. Comparing actual labels and prediction labels for the validation data set.</i>
</p>

The top 3 features based on importance were: moving_avg_sum_cnt (0.07162572118271372), moving_avg_sst (0.06833591323101115), rolling_stddev_sss (0.06738219203805107)

The bottom 3 features based on importance were: yoy_change_avg_sss (0.03811226434480753), yoy_change_sum_cnt (0.037122476967009446), yoy_dif_long (0.02880474717987468)

<p align="center">
  <img src="https://github.com/sanrajagopal/DSC232RGroupProject/blob/af4b4c7ef8eead00d05631c9e60773087de6ff32/pics/model1b3.png" alt="Model 1b - Random Forest v2 Importance Features">
</p>
<p align="center">
  <i>Figure 11. Model 1b - Random Forest v2 Importance Features. Importance features extracted from the model ranked in descending order.</i>
</p>

### Model 2 - Gradient Boost 

Using the same data as in Model 1b, hyperparameter tuning via grid search found the best parameters to be a depth of 5, a max iteration of 150, and a step size of 0.1. Our fitting graph mirrored this somewhat but showed our model overfitting.

<p align="center">
  <img src="https://github.com/sanrajagopal/DSC232RGroupProject/blob/af4b4c7ef8eead00d05631c9e60773087de6ff32/pics/model21.png" alt="Model 2 - Gradient Boost Fitting Graph">
</p>
<p align="center">
  <i>Figure 12. Model 2 - Gradient Boost Fitting Graph. Comparing the prediction error rate of the GBT Classifier model between the training and validation data sets over increasingly complex models (increasing maxDepth hyperparameter values) to check for model fit.</i>
</p>

Training the GBT classifier model on the same data used in model 1b produced the following errors:

Train Error: 0.019033117624666973

Validation Error: 0.31543624161073824

Ground testing and a confusion matrix showed semi-accurate predictions of “unsafe” species (safe?=0) and ”safe” species (safe?=1)

<p align="center">
  <img src="https://github.com/sanrajagopal/DSC232RGroupProject/blob/af4b4c7ef8eead00d05631c9e60773087de6ff32/pics/model22.png" alt="Model 2 - Gradient Boost v2 Confusion Matrix">
</p>
<p align="center">
  <i>Figure 13. Model 2 - Gradient Boost Confusion Matrix. Comparing actual labels and prediction labels for the validation data set.</i>
</p>

The top 3 features based on importance were: avg_shoredistance (0.07222773109597815), rolling_stddev_sss (0.06309474509744703), avg_bathymetry (0.061783930675519315)

The bottom 3 features based on importance were: yoy_change_avg_sss (0.039985498358066654), yoy_dif_lat (0.038735048225218095), yoy_change_sum_cnt (0.02733645994338745)

<p align="center">
  <img src="https://github.com/sanrajagopal/DSC232RGroupProject/blob/af4b4c7ef8eead00d05631c9e60773087de6ff32/pics/model23.png" alt="Model 2 - Gradient Boost Importance Features">
</p>
<p align="center">
  <i>Figure 14. Model 2 - Gradient Boost Importance Features. Importance features extracted from the model ranked in descending order.</i>
</p>

### Final Model - Random Forest v2

Based on the results of the validation accuracy and fitting graphs, the final model was selected to be the adjusted random forest model. Trained on the combined training and validation data we achieved the following error results:

Train Error: 0.10430148458317468

Test Error: 0.32366589327146167

Similar to the original altered random forest model, ground testing and a confusion matrix showed semi-accurate predictions of “unsafe” species (safe?=0) and ”safe” species (safe?=1)

<p align="center">
  <img src="https://github.com/sanrajagopal/DSC232RGroupProject/blob/af4b4c7ef8eead00d05631c9e60773087de6ff32/pics/final1.png" alt="Final Model - Random Forest v2 Confusion Matrix">
</p>
<p align="center">
  <i>Figure 15. Model 1b - Random Forest v2 Confusion Matrix. Comparing actual labels and prediction labels for the test data set.</i>
</p>

The top 3 features based on importance were: moving_avg_sst (0.07491767966419666), avg_shoredistance (0.07251469520331419), moving_avg_sum_cnt (0.06469008872862284)

The bottom 3 features based on importance were: yoy_change_sum_cnt (0.037076837442903465), yoy_dif_long (0.03255308429305591), yoy_dif_lat (0.03173588684044194)

<p align="center">
  <img src="https://github.com/sanrajagopal/DSC232RGroupProject/blob/af4b4c7ef8eead00d05631c9e60773087de6ff32/pics/final2.png" alt="Final Model - Random Forest v2 Importance Features">
</p>
<p align="center">
  <i>Figure 16. Final Model - Random Forest v2 Importance Features. Importance features extracted from the model ranked in descending order.</i>
</p>

## Discussion Section 

### Data Exploration

For our data exploration, we wanted to be familiar with the dataset and its many features. Since there were an overwhelming number of columns (over 200) we decided to focus on learning which features would be most important for our prediction models.

By looking at the number of observations per year, we noticed that there were insufficient and inconsistent numbers of observations before 1900. This informed us that we would benefit from removing all data whose year was before that. As our concern was that of entries whose year was on or after the assessment year found in the IUCN dataset, this did not impact our models.

We also noticed a huge spike of observations in 2016, which was clarified by the OBIS data manager to be due to the ingestion of a particularly large DNA dataset that year. Luckily this matched up with what we saw plotting Orcinus orca (Orca aka Killer Whale) observations over the years, lending credence to this being species agnostic. While an attempt was made to normalize 2016 in pre-processing, those efforts proved to degrade the performance of our model.

Looking at the red list categories per species directly, we noticed that some of the species may have outdated red list classifications or none at all. LR (Lower Risk) was discontinued in 2001 as a classification, as well as all of its derivatives. While still in use for species that haven’t been assessed since 2001, it indicated to us that the OBIS dataset may have out-of-date, or missing information. Additionally, as stated in the results, the OBIS dataset possessed approximately 2000 unique species names and red list category pairs. On the IUCN website, marine species alone account for over 23,000 assessments. This fact helped solidify our decision to introduce the IUCN dataset to help us fill in the gaps of missing or outdated data. 

Finally, null values. As we narrowed down the list of usable features, this did cut down on the null values we needed to be concerned with. However, due to our heavy usage of aggregate values later in preprocessing, those that remained required special attention. The reason for such prevalent null values was 2 fold: species identity and abandoned columns. Due to the fact a species identity often wasn’t known, only a higher taxon was provided by the user. But even when an organism was identified, organisms rarely occupy all hierarchies of taxonomic data, as such those taxonomic categories were left as null. This coupled with the fact that the OBIS dataset has been expanded on and changed over the years, makes many columns redundant.

### Preprocessing

#### OBIS Dataset

From our data exploration, we were able to determine which columns were needed in predicting red list categories. The columns that were kept included data on location, temperature, environmental features, and taxonomic names for species. However, information on dates was our first challenge. We noticed that multiple columns were related to a date. To keep the data more uniform, we decided to use the year column which had been cleaned by OBIS quality control. However, if it was null we would use another column called eventDate and extract the year data from there instead. While this helped clean the data, a more thorough method might have achieved a larger return.

While we explored the idea of bringing in more red list category data, we found an interesting roadblock born from taxonomic re-classification. Taenioides jacksoni (aka the bearded goby) was renamed Trypauchenopsis intermedia sometime in the 1960s, and this was reflected in our data. To correct it we merely performed a string replacement, but it is unknown how many species have gone through similar changes. While taxonomic name changes are not unusual, this was the only one we found. Luckily this was an easy fix as we just had to rename the species name to be consistent.

To better ensure the usage of accurate data, we utilized a column called coordinateUncertainty. As the name suggests it was a column providing the accuracy of the point of observation. To maintain integrity, we decided to only keep measurements that were less than 1000 meters. We also transformed any null uncertainty values to 0 as there is a requirement for users to enter their coordinates, but not their coordinate estimations. A lack of uncertainty we took as a form of reasonable certainty.

Next, we needed to better quantify individual counts as described in each entry. During our exploration, we noticed 2 columns, individualCount, and organismQuantity. Each provided some amount of information regarding the counts of organisms seen per entry, but organismQuantity was paired with organismQuantityType which indicated that the values in organismQuantity could range from actual counts to densities in a given area. Given that fact, we stuck with individualCount to provide the counts for any entry it was provided, assuming a count of 1 for any entry it was not. A deeper analysis of organismQuantity and organismQuantityType could reveal better estimations of sighting numbers in future analysis.

As noted from data exploration, the counts for each year were heavily skewed as the years went on, especially for 2016. We decided to use a log transformation to standardize the counts and add more weight to years that had lower count totals. This later proved detrimental to our models and better performance was seen when this was removed.

Since we were trying to look at time series data between species, we needed to aggregate the numerical data from our dataset to help with predictions for the model. Using species name and year as the grouping criteria, we obtained average temperature, average salinity, average shore distance, min/max of latitude and longitude, average bathymetry, and the total counts of observation per year for each species. This information we would later use to compute more temporal data.

Throughout these preprocessing steps, we periodically had to remove null values to make sure our data was able to be used by the prediction models of our choosing. The most important columns for our analysis were environmental data related which was consistently provided by OBIS, but still had millions of missing entries. While some of it could be retrieved or assumed, as stated in the data exploration discussion section, some data was simply not provided. While accompanying data such as NOAA buoy sampling to incorporate rolling oceanographic chemical data, this proved to be beyond the scope of this project. So ultimately, by removing these null values, our dataset was heavily reduced in volume. 

#### IUCN Dataset

To incorporate the IUCN dataset it was necessary to first clean the provided csv file as the IUCN dataset was littered with comments and unformatted text columns. Rigorous manual cleaning was applied, with any further errors being dropped during the joining process.

With the new IUCN dataset handling null and outdated classifications, we wanted to create a binary column for easier classification. This was labeled as safe or unsafe (true or false which was later changed to 0 or 1). A species was labeled as safe if it was near threatened or least concerned and not safe otherwise. However, we came to realize that leaving the classifications as the original 7 may have been better for the model. This fact was unfortunately never explored but might prove interesting as a follow-up to this project.

#### Merged Dataset 

For our final pre-processing step we generated time series data. We included moving averages, lag 1 data, and year-on-year changes for multiple numerical variables. We kept our moving average to a range of 3 years due to the limited scope of available data after removing anything older than the assessment years in our IUCN dataset. We did not look into lags more than 1 or multiple-year or year changes which could have altered the accuracy of our model. We initially did not standardize any values besides sum counts, which contributed to an inaccurate performing first model. However, we included standardized values for later models.

### Model 1a - Random Forest

We chose a random forest because the ensemble method can better utilize multiple features that a decision tree would not be able to capture. Being able to better generalize certain relationships between these features would ideally outperform a singular tree model.  

Due to the random forest classification model specifications, we had to remove many columns that were not numerical or binary. This included many of the columns describing the taxon of species such as kingdom, domain, etc. These columns could have helped with the accuracy and prediction but we could not have tested this. It is possible that word embeddings may have been applied and should be considered for future applications. One-hot encoding would have caused the dimensionality to explode and would not be recommended. We also did not scale other numerical features which we later did for future models.

Looking at the validation errors the model appears to be doing extremely well. However, the model is very biased because the data has more safe labels than unsafe labels. The ratio of correctly predicted labels merely mirrored our data distribution showing that our model was not training correctly. To have a model that generalizes better we decided that we would need to reduce the number of safe species to fit 50% of the total data. In addition, we would reduce the number of features followed by a new grid-search and fitting graph which would provide a model that better generalizes to unseen data. 

### Model 1b - Random Forest v2

We decided to do another random forest model with the adjustments to the data (see discussion). With these adjustments, the model better generalized to the data but our accuracy suffered greatly. However, this trade-off was worth it as it could more accurately identify at-risk species as opposed to blanket classifying everything as safe. 

Additionally, certain features started to become more relevant. Moving average sst (sea surface temperature) being a top feature could indicate that organisms in areas where climate change has a larger impact on sst (such as polar regions and tropical) might be facing the greatest threat. Moving average sum count is self-explanatory as those numbers increasing and decreasing would have a direct correlation on population counts. But the rolling standard deviation sea surface salinity being an impactful feature is hard to reason out. It could be that organisms with narrow salinity tolerances are more impacted by higher variability in their local salinity, similar to organisms being affected by changing sst, and this is reflected in the model. But this would require deeper analysis. Regardless, we wanted to try a different model that could potentially have better accuracy given the same adjustments we made. 

### Model 2 - Gradient Boosted Trees

We chose gradient-boosted trees as the second model as they could potentially handle outlying errors better than a random forest by iteratively correcting errors from previous weak learners. The random forest model struggled with many outliers, which we hoped gradient-boosted trees would manage more effectively. Gradient-boosted trees also tend to perform better on complex data, and given that our data is mostly environmental factors that have varied interactions over time, we expected gradient-boosted trees to perform better.

The gradient-boosted tree model showed similar accuracy to the model 1b. However, given that the run time for this model is much higher than random forest, it may not be worth the slight difference in accuracy. Additionally, the fitting graph showed our model had overfitted in nearly all levels of depth, which might mean that the model would have difficulty in generalizing to new data. 

Despite this, some interesting results were gleaned from the feature importance extraction. Consistently, sst and sss (sea surface salinity) were among the top features in determining species red list category classification, very similar to the random forest models. Average shore distance being the most influential feature could be explained by human impact as organisms closer to shore might have more human interactions. Or it could be a bias in the data as those reporting this data are going to use observations they have the easiest access to. Bathymetry might have a similar story in that as a majority of marine species live in shallow waters to benefit from the phytoplankton requiring sunlight, they are also the easiest to access by humans. This, in turn, could make them vulnerable to human activities, or it could make them easier to catalog, making them part of a skewed sampling process.

While the GBT classifier did not meet expectations, it did provide interesting insights into marine ecological relationships that would benefit from further analysis.

### Final Model - Random Forest v2

After considering the models and their similar performances, we decided to use the altered random forest model as our final model. Its fitting graph showed much more promise compared to the other models, it had a similar if not equal performance to the GBT model, as well as being able to be trained faster. These qualities made it the best candidate for our final model. While its test performance was not at the level we were hoping for, further feature expansion may improve performance in future iterations.

As seen in the prior models, sst and sss were consistently the most important features throughout the dataset. What was surprising was how spread counts were in their influence on the final model, with rolling averages being the highest impact. Year-over-year features (denoted yoy in the variable names), were consistently the least influential temporal features, which does make some sense as a species change from one year to the next is not representative of a larger trend. Additionally, location information (denoted lat and long in the variable names), was also a poorer predictor than expected. There was some anticipation that changing organism ranges might have a relationship with red list category classification. But that might have been a stretch to assume when such information might have been sourced from captive animal locations. As environmental factors seemed to play the largest role, using temporally dynamic environmental indicators such as dissolved oxygen, chlorophyll-a, or nitrogen levels (estimated from satellite imagery) could provide further improvements to our models.

## Conclusion

Models that focus on how marine biodiversity changes over time can help inform policymakers and scientists on how to make decisions about topics like fishery availability and conservation effort allocation. Expanding these models to data sets like GBIF (Global Biodiversity Information Facility) could push this idea beyond marine ecosystems, helping understand the ongoing effects of climate change and human impact on our environment.

However, this predictive model is only a first step. An accuracy of 68% is not reliable enough to generalize to other populations. One limitation we faced was that our IUCN dataset only possessed the current classification of a given species. Expanding that to include older assessments of red list category classification might provide insights into long-standing trends that could improve our model performance. Further data expansion could be done by merging oceanographic chemical data collected by NOAA to provide further indicators of a species' changing status.

Our models also employed a binary classification. Classifying the full range of red list categories would be much more useful to an end user who might want a finer grain understanding of a species standing. Additionally, while we tested our model random forest and GBT classification, using more complex models such as neural networks or stacking various models may utilize hidden relationships that our current models might not be able to rely on. 

Finally, it is important to recognize that conflating global marine species' environmental interactions with local fauna might not be an accurate reflection of the complex interactions happening within a select biome. Using this framework to transform local data on a case-by-case might provide a better understanding of how a species is reacting to changes in the environment over time. Careful consideration must be taken of how we can use machine learning models to help species conservation.

## Collaboration

Ekai Richards: Co-author and co-coder.
- Contributed to both the code and write-up.
- Worked as a team. 
- Set up meetings

Sanjana Rajagopal: Co-author and co-coder. 
- Contributed to both the code and write-up. 
- Worked as a team. 
- Set up meetings
