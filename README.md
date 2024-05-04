# DSC232RGroupProject

## Introduction
The OBIS dataset is a collection of marine animal observations built from collected documentation, volunteer-provided observations, and aggregated data sets whose records go back over 100 years.
This open-access data provides information on marine bio-diversity to help highlight insights for scientific communities and generate potential sustainability practices.
Our goal is to dive into the data set and pull out some insights regarding how the biodiversity of marine life has been changing.

## Data Exploration
We explored several features in the data set; firstly to determine the number of counts per year. We quickly discovered that there were several entries whose data managed to avoid the OBIS team's QC, having erroneous years.
To avoid large skews (as observations increased over the years) as well as handling data points that may have incorrect dates, we limited the data to only include years greater than 1900.
We additionally noticed a large spike in 2016 as the number of observations was about 10-fold greater as compared to neighboring years.
This was reflected in our exploration of Orca sightings as they seemed to mirror the overall count graph shape which indicates a general increase in observations rather than a particular bloom being the cause.

As we are mostly concerned with changing biodiversities, we deemed it important to find out how much of the data has been appropriately labeled with IUCN's redlist categories.
We found that about 10% of entries had actual redlist categories listed in their entries, with approximately 2000 unique species with IUCN categories.
This showed this data set was not exhaustive with that information, IUCN has over 150,000 species they've analyzed with over 40,000 currently threatened so there is a clear disparity.
As an example, Atlantic tuna had gone from Endangered to Least Concern in 2011, neither of which was categorized in this data. If this angle is to be examined it may be pertinent to incorporate this information.

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
The dataset is quite large so we will need to clean the data in several ways.
1) Remove irrelevant columns - Numerous columns show duplicate information, do not have a variable description provided, or do not provide useful insight, so a majority of these columns will need to be removed.
2) Remove missing data - A number of entries are corrupted or do not have useable information which need to be removed
3) Standardizing - Due to the nature of human observation some standardization will need to be done regarding names via changing entries to lower-case and removing whitespace
4) Standardizing unknown values - Convert all blank, null, or n/a values into a singular 'unknown' type
5) Standardize the redlist_category - There are features of the redlist that are outdated and should be updated to current standards
6) Scale to observation counts - The number of observations varies, especially comparing older years to recent ones. Normalizing counts based on the counts made that year could help to normalize the data.
7) Potential feature expansion - Depending on the needs of our investigation we may use other data sets from NOAA such as WOA to expand our feature list
