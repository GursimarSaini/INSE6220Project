# PCA and Binary Classification on Office Room Occupancy
---------------------------------------------------------
---------------------------------------------------------
## Abstract:
> Principal Component Analysis (PCA) is a fast and flexible unsupervised dimensionality reduction method that transforms a high dimensional data with correlated features to low dimenesional data with uncorrelated features. This report illustrates the use of PCA when applied to the office room occupancy data set attributes to classifiy if the room is occupied. Determination of occupancy detection in a room can lead to considerable energy savings in modern smart home/buildings. **~~Still some part remaining~~**

> Keywords- Principal Component Analysis (PCA), Binary Classification, 
---------------------------------------------------------
## Ⅰ. Introduction:

With the decreasing price of sensors and the availability of reasonable computational power for automation systems, determining occupancy is a very promising way to lowering energy usage in buildings through appropriate control of HVAC and lighting systems. Threat of climate adversity has made it important for the production of most energy efficient products [1]. The precise detection of occupancy in buildings has lately been projected to save energy in the range of 30 to 42 percent. When occupancy data was employed as an input for HVAC control algorithms, it resulted in energy savings of 37 percent withour sacrificing indoor climate and between 29 and 80 percent in another [2]. When privacy matters are considered, it makes much more sense to use sensors for getting accurate occupant numbers than to use cameras. Determining building inhabitants behavior and Security are another two applications for occupancy detection. 

The research [2] used data from light, temperature, humidity, and CO2 sensors to detect occupancy, as well as a digital camera to determine ground occupancy for data labelling. This data set created for occupancy detection is used for this study. 

Working with a huge dataset as what used in this study is usually perplexing and laborious. To make the research easier, the approach must incorporate dimension reduction, while preserving the majority of the data variability. PCA is generally used for such tasks [3], which is described in Section 3 and its implementation showcased in Section 5, after giving a brief Exploratory Data Anlaysis in Section 2. Section 4, throws light on applicable Classifiers with brief explanations and discussion of the most promising model that helps in detection for this process in Section 6. Section 7 closes with a summary of the findings.

---------------------------------------------------------
## Ⅱ. Exploratory Data Analysis:

Exploratory Data Analysis refers to the crucial process of conducting preliminary investigations on data in order to uncover patterns, spot anomalies, test hypotheses, and validate assumptions using summary statistics and graphical representations. Here it's done in three parts, first by giving a brief introduction for the raw data set, then discussing the cleaning process and description for used data set. At last, checking distribution and outliers with Box Plots and correlation of points with Correlation Matrix.

#### Raw Data Set Description

As mentioned in the study [2], the following variables were observed in an office space with approximate dimensions of 5.85m, 3.50m, 3.53m (W D H): timestamp, temperature, humidity, light, and CO2 levels. The study collects the data using a microcontroller. It was linked to a ZigBee radio, which was used to relay the data to a recording station. A digital camera was utilised to assess whether or not the room was inhabited. Every minute, the camera time stamped an image, which was then manually examined to identify the data. The humidity ratio is another additional variable in the data model, calculated as ( W = 0.622*(pw/(p-pw)) ).

The data was collected in February in Mons, Belgium, during the winter. The room was heated by hot water radiators, which kept the temperature above 19 degrees Celsius. The models are tested for data sets with the office door open and closed in order to estimate the difference in occupancy detection accuracy provided by the models. The measurements were obtained at 14-second intervals/3-4 times every minute, and then averaged for that minute.

#### Data Cleaning

All three data sets were missing column name for their first column, which was named as "id" and then dropped in data pre-processing. For the purpose of this study, only one test data 1 and training data will be used. 

#### Used Data Set Description

The description for these two datasets is summarized in **Table 1**. No duplicate rows or NaN values were found for both of the datasets. And all the values are floating point numbers, except the column "Occupancy" which is labelled with int values, 0 and 1 

![Table 1: Data Set Description](figures/tables/DataSetDescription.png "Table 1: Data Set Description")

The distribution of class can be seen with the bar plot in **Figure 1**. As said, the label 1 represents that the room was occupied and Class 0 for unoccupied rooms.

![Figure 1: Bar Plot for Classes](figures/classBarPlot.png "Figure 1: Bar Plot for Classes")

#### Data Analysis

Standardization is put into use to adjust each input variable independently by removing the mean (called centering) and dividing by the standard deviation to shift the distribution to have a mean of zero and a standard deviation of one [4]. After standardization, the Descriptive Statistics metrics can be seen in **Table 2**.

![Table 2: Descriptive Statistics](figures/tables/DescriptiveStatistics.png "Table 2: Descriptive Statistics")


---------------------------------------------------------
## Ⅲ. Principal Component Analysis:

PCA is typically used to reduce the dimensionality of data while retaining as much of the information present in the original data as feasible. It does this by examining a data table including observations characterised by numerous dependent variables that are, in general, inter-correlated. Its purpose is to extract the key information from the data table and express this information as a set of new orthogonal variables known as principal components. Simply put, PCA is important so as to:

1. Extract the most relevant information from the data table, 
2. Compress the size of the data set by maintaining just the most significant information, 
3. Simplify the data set description, and 
4. Evaluate the structure of the observations and variables.

PCA output comprises of coefficients that specify the linear combinations used to obtain the new variables (PC loadings) as well as the new variables themselves (PCs). The first PC must have the greatest potential variance. The second component is calculated with the constraint of being orthogonal to the first component and having the greatest possible inertia. The other components are calculated in the same way. [5]

#### Steps for PCA algorithm:
The data should be structured in a typical matrix format, with n rows of samples and p columns of variables. There should be no missing values: each variable should have a value for each sample, which can be zero. [3],[6]
1. Centering the dataset: The first step is centering of data on the means of each variable, which is done by subtracting the mean of a variable from all of its values. This ensures that the data cloud is centered on the origin of our main components, but it has no effect on the spatial connections of the data or the variances along our variables.

---------------------------------------------------------
## Ⅳ. Classification Algorithms:

---------------------------------------------------------
## Ⅴ. PCA Results:

---------------------------------------------------------
## Ⅵ. Classification Results:

---------------------------------------------------------
## Ⅶ. Conclusions:

---------------------------------------------------------
## Ⅷ. References:
[1] Boardman, B. (2004). New Directions for Household Energy Efficiency: Evidence from the UK. Energy Policy, 32(17), 1921–1933. https://doi.org/10.1016/j.enpol.2004.03.021 
[2] Candanedo Ibarra, Luis & Feldheim, Veronique. (2015). Accurate occupancy detection of an office room from light, temperature, humidity and CO2 measurements using statistical learning models. Energy and Buildings. 112. 10.1016/j.enbuild.2015.11.071.
[3] A. Ben Hamza, Advanced Statistical Approaches to Quality, unpublished.
[4] https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
[5] Principal Component Analysis Herv´e Abdi · Lynne J. Williams
[6] https://strata.uga.edu/8370/lecturenotes/principalComponents.html


 

---------------------------------------------------------
---------------------------------------------------------
---------------------------------------------------------
## Ⅸ. Abstract:
>
---------------------------------------------------------
## Ⅹ. Abstract:
>
---------------------------------------------------------

