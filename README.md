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

Working with a huge dataset as what used in this study is usually perplexing and laborious. To make the research easier, the approach must incorporate dimension reduction, while preserving the majority of the data variability. PCA is generally used for such tasks [3], which is described and implemented in Section 3, after giving a brief Exploratory Data Anlaysis in Section 2. Section 4, throws light on applicable Classifiers with brief explanations and discussion of the most promising model that helps in detection for this process in Section 5. Section 6 closes with a summary of the findings.

---------------------------------------------------------
## Ⅱ. Exploratory Data Analysis:

Exploratory Data Analysis refers to the crucial process of conducting preliminary investigations on data in order to uncover patterns, spot anomalies, test hypotheses, and validate assumptions using summary statistics and graphical representations. Here it's done in three parts, first by giving a brief introduction for the raw data set, then discussing the cleaning process and description for used data set. At last, checking distribution and outliers with Box Plots and correlation of points with Correlation Matrix.

#### Raw Data Set Description

As mentioned in the study [2], the following variables were observed in an office space with approximate dimensions of 5.85m, 3.50m, 3.53m (W D H): timestamp, temperature, humidity, light, and CO2 levels. The study collects the data using a microcontroller. It was linked to a ZigBee radio, which was used to relay the data to a recording station. A digital camera was utilised to assess whether or not the room was inhabited. Every minute, the camera time stamped an image, which was then manually examined to identify the data. The humidity ratio is another additional variable in the data model, calculated as ( W = 0.622×(pw/(p-pw)) ).

The data was collected in February in Mons, Belgium, during the winter. The room was heated by hot water radiators, which kept the temperature above 19 degrees Celsius. The models are tested for data sets with the office door open and closed in order to estimate the difference in occupancy detection accuracy provided by the models. The measurements were obtained at 14-second intervals/3-4 times every minute, and then averaged for that minute.

#### Data Cleaning

All three data sets were missing column name for their first column, which was named as "id" and then dropped in data pre-processing. For the purpose of this study, only one test data 1 and training data will be used. 

#### Used Data Set Description

The description for these two datasets is summarized in **Table 1**. No duplicate rows or NaN values were found for both of the datasets. And all the values are floating point numbers, except the column "Occupancy" which is labelled with int values, 0 and 1 

![Table 1: Data Set Description](figures/tables/DataSetDescription.png "Table 1: Data Set Description")

The distribution of class can be seen with the bar plot in **Figure 1**. As said, the label 1 represents that the room was occupied and Class 0 for unoccupied rooms.

![Figure 1: Bar Plot for Classes](figures/classBarPlot.png "Figure 1: Bar Plot for Classes")

#### Data Analysis

Standardization is put into use to adjust each input variable independently by removing the mean, and dividing by the standard deviation to shift the distribution to have a mean of zero and a standard deviation of one [4]. After standardization, the Descriptive Statistics metrics can be seen in **Table 2**. If it's not done, then covariances for larger number ranges will be much higher. 

![Table 2: Descriptive Statistics](figures/tables/DescriptiveStatistics.png "Table 2: Descriptive Statistics")

Putting this standardize matrix in a Box Plot gives us the idea about the distribution of the data, measures of central tendency and spread. In **Figure 2**, we can see that, all the data attributes are positively skewed to an extent. Data is centered around 0, beacuse of standardization and variability is minimum for the same reason. Outliers are present for 3 of the 5 attributes and all of them are on the skewed side of whiskers.

![Figure 2: Box Plot](figures/boxPlot.png "Figure 2: Box Plot")

To understand the relationship among the attributes, Correlation Matrix and Pair Plot are used. It's evident that almost all of the parameters are positively correlated with each others. There is no presence of any variable with negative correlation with all of the others variables. CO2 has significant correlation with rest of variables over others as seen in **Figure 3**

![Figure 3: Correlation Matrix](figures/CorrelationMatrix.png "Figure 3: Correlation Matrix")

This can also be supported with the help of pairplot in **Figure 4**. The strong positve correlations are determined with increasing line. Whereas, weak correlations form clusters rather than an increasing line in pair plot. 

![Figure 4: Pair Plot](figures/pairPlot.png "Figure 4: Pair Plot")

The multiple correlations among data set parameters, is the reason why PCA is implemented to get un-correlated data.

---------------------------------------------------------
## Ⅲ. Principal Component Analysis:

PCA is typically used to reduce the dimensionality of data while retaining as much of the information present in the original data as feasible. It does this by examining a data table including observations characterised by numerous dependent variables that are, in general, inter-correlated. Its purpose is to extract the key information from the data table and express this information as a set of new orthogonal variables known as principal components. Simply put, PCA is important so as to:

1. Extract the most relevant information from the data table, 
2. Compress the size of the data set by maintaining just the most significant information, 
3. Simplify the data set description, and 
4. Evaluate the structure of the observations and variables.

PCA output comprises of coefficients that specify the linear combinations used to obtain the new variables (PC loadings) as well as the new variables themselves (PCs). The first PC must have the greatest potential variance. The second component is calculated with the constraint of being orthogonal to the first component and having the greatest possible inertia. The other components are calculated in the same way. [6]

#### Steps for PCA algorithm:

We need to make sure, data should be structured in a typical matrix format, with n rows of samples and p columns of variables. There should be no missing values: each variable should have a value for each sample, which can be zero [7]. The steps needed for PCA is as follows [3].
1. Centering the dataset: For this step we substract the mean of a variable from all of its values, so that the data stays centered on the origin of main components, because any algorithm which is based on distance computations are affected a lot if the data used isn't normalized/centeralised [8]. 

Y = HX .......... (1)

For our study, standardisation is preferred over centering to avoid precision error when range of variables is different.

![Table 3: Original Data](figures/tables/ogData.png "Table 3: Original Data")

![Table 4: Standardized Data](figures/tables/stanData.png "Table 4: Standardized Data")

2. Calculate Covariance Matrix: Covariance matrix of size p×p is produced to check if data set has correlated features and also so as eigen decomposition can be applied to the data [3].

S = (1/n-1) × Y<sup>T</sup> × Y .......... (2)

![Table 5: Covariance Matrix](figures/tables/stanData.png "Table 5: Covariance Matrix")

The study notebook doesn't include this step for PCA implementation, as PCA library directly outputs the PC scores.

3. Eigen Decomposition: Eigenvectors and eigenvalues are obtained from the Covariance matrix S with eigen dencomposition. Eigenvectors give direction of Principal Components with variance of PCs dented with eigenvalues and are given by [3]:

S = A×λ×A<sup>T</sup> .......... (3)

where A is a p×p orthogonal eigenvector matrix and is a diagonal eigenvalue matrix.

In our case, we have a 5×5 eigenvector matrix and a 1×5 column of eigenvalues, both given in **Figure 5** and **Figure 6** respectively.

![Figure 5: eigenvectors](figures/eigenvectors.png "Figure 5: eigenvectors")

![Figure 6: eigenvalues](figures/eigenvalues.png "Figure 6: eigenvalues")

4. Principal Components: The last step yields a n×p matrix Z, with its rows giving the observed values and columns representing the PCs. Given by equation [3]:

Z = Y×A .......... (4)

The variance of jth PC is given as following [3]:

![Figure 7: Explained Variance](figures/explainedVariance.png "Figure 7: Explained Variance")

where λ<sub>j</sub> gives the variance of jth PC. Both Scree/Elbow plots can be used to get an idea of how many PCs are needed to represent the variance present in the data. In this study, we found out that variance accounted for first PC is l<sub>1</sub> = 54.7% and by 2nd PC it is l<sub>2</sub> = 33.9% and that by 3rd PC is l<sub>3</sub> = 6.97%. The elbow joint in the scree plot, shows a bend at PC number 3, that is also supported with Pareto Chart. So, it's safe to assume that dimensions of eigenvector or Z components can be reduced to 3.

![Figure 8: Scree Plot](figures/screePlot.png "Figure 8: Scree Plot")

![Figure 9: Pareto Chart](figures/paretoChart.png "Figure 9: Pareto Chart")

The **Table 6** table gives a subset of the PCs, 8143×5 as 8143×3 because first 3 PCs explain 99.98% of the whole variance of dataset. The first PC component Z<sub>1</sub> is given by [3]:

![Table 6: PCs](figures/tables/pcs.png "Table 6: PCs")  .......... (5)

Z<sub>1</sub> = 0.55007×X<sub>4</sub> + 0.501116×X<sub>5</sub> + 0.414149×X<sub>3</sub> + 0.395664×X<sub>2</sub> + 0.343856×X<sub>1</sub>

X<sub>4</sub> (CO2), X<sub>5</sub> (Humidity Ratio), X<sub>3</sub> (Light), X<sub>2</sub> (Humidity), and X<sub>1</sub> (Temperature), contribute most to the 1st PC, respectively. For Z<sub>2</sub>, we got = 

- 0.574111×X<sub>2</sub> + 0.535864×X<sub>1</sub> + 0.444612×X<sub>3</sub> - 0.413692×X<sub>5</sub>  + 0.120106×X<sub>4</sub>

For both of the first two principal components we don't have any attribiute contributing negligibly to them. But in case of, third principal component, X<sub>2</sub> doesn't affect it that much, so effective PC will be:

Z<sub>3</sub> = - 0.713374×X<sub>1</sub> + 0.665424×X<sub>3</sub> - 0.189517×X<sub>5</sub> + 0.110938×X<sub>4</sub>   

This same could be verified with the help of PC coefficient Plot as in **Figure 10**. Temperature and Humidity lies in the lower range for the first PC, along with CO2 and Humidity Ration being the most important factors for consideration, with light being somewhere in the middle of first two and latter two. Whereas, all the bottom 3 contributors of 1st PC got to be at the top for 2nd PC and C02 being the least.

![Figure 10: PCcoefficient](figures/pcCoefficient.png "Figure 10: PCcoefficient")

BiPlot gives the same information as of **Figure 10**. The angles between the vectors (rows of eigenvector matrix) and axes (representing the first two PCs) gives the contribution of variables to PCs [3], i.e., the vector with smallest angle with the axe contributes most to that axe/PC. Also, each observation is scattered as a point in the plot.

![Figure 11: BiPlot](figures/biplot.png "Figure 11: BiPlot")

This same could be represented for three PCs with help of 3d BiPlot as shown in **Figure 12**.

![Figure 12: BiPlot3d](figures/biplot3d.png "Figure 12: BiPlot3d")

The final step in PCA is to inspect the control chart to ensure that all of the points are between the Upper Control Limit (UCL) and Lower Control Limit (LCL) (LCL). Equation 6 gives a comprehensive guide for these limits [3]:

![Figure 13: controlEquations](figures/controlEquations.png "Figure 13: controlEquations")

where λ<sub>j</sub> is the eigenvalue for the j<sup>j</sup> PC.

---------------------------------------------------------
## Ⅳ. Classification Algorithms:

---------------------------------------------------------
## Ⅴ. Classification Results:

---------------------------------------------------------
## Ⅵ. Conclusions:

---------------------------------------------------------
## Ⅶ. References:
[1] Boardman, B. (2004). New Directions for Household Energy Efficiency: Evidence from the UK. Energy Policy, 32(17), 1921–1933. https://doi.org/10.1016/j.enpol.2004.03.021 
[2] Candanedo Ibarra, Luis & Feldheim, Veronique. (2015). Accurate occupancy detection of an office room from light, temperature, humidity and CO2 measurements using statistical learning models. Energy and Buildings. 112. 10.1016/j.enbuild.2015.11.071.
[3] A. Ben Hamza, Advanced Statistical Approaches to Quality, unpublished.
[4] https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
[5] https://godzillabutnicer.com/the-why-of-principal-component-analysis-standardization-covariance/
[6] Principal Component Analysis Herv´e Abdi · Lynne J. Williams
[7] https://strata.uga.edu/8370/lecturenotes/principalComponents.html
[8] https://datafai.com/2017/10/27/data-standardization-or-normalization/

 

---------------------------------------------------------
---------------------------------------------------------
---------------------------------------------------------
## Ⅸ. Abstract:
>
---------------------------------------------------------
## Ⅹ. Abstract:
>
---------------------------------------------------------

