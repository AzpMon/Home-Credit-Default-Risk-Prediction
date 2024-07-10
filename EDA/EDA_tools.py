import seaborn as sns
import matplotlib.pyplot as plt
from pyspark.sql import SparkSession
from pyspark.sql.types import IntegerType, FloatType, StringType
from pyspark.sql.functions import col, avg, sum, when
from phik import phik_matrix
import numpy as np
import pandas as pd
from IPython.display import display, HTML
from matplotlib.colors import LinearSegmentedColormap
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', None)







# Seaborn configuration
sns.set_style("darkgrid")

class EDAtools:
    def __init__(self, spark):
        """
        Initializes the class with a Spark session.

        Args:
            spark (SparkSession): Spark session.
        """
        self.spark = spark

    # Para eliminar el caché junto a un dataframe que ya no se utilizará 
    def borrar(self,dataframe):
        """
        Función para eliminar tanto el caché como un dataframe que ya no se utilizará 
        Args:
            datagrame (DataFrame): Dataframe a eliminar
        """
        dataframe.unpersist()
        del dataframe
    
    def read_dataframe(self, csv, header=True):
        """
        Reads a DataFrame from a CSV file.

        Args:
            path (str): Path to the CSV file.
            header (bool, optional): Indicates if the CSV file has headers. Default is True.

        Returns:
            DataFrame: DataFrame read from the CSV file.
        """
        df = self.spark.read.option('header','true').csv(csv)
        return df

    def show(dself, dataframe, limit = 5):
        """
        Función que muestra el dataframe (de pyspark) en el formato de Pandas con todas las columnas
        limitandose limit columnas. Se elimina el dataframe que se utiliza.
    
        Args:
            dataframe (pyspark.sql.dataframe.DataFrame) : DataFrame a mostrar
            limit (int) : Límite de registros a imprimir, por default 5
        
        """
        pandasDataFrame__ = dataframe.limit(limit).toPandas()
        display(pandasDataFrame__)
        
        del pandasDataFrame__

    def join_dataframes(self, df1, df2, join_column, join_type='left'):
        """
        Joins two DataFrames based on a common column.

        Args:
            df1 (DataFrame): First DataFrame.
            df2 (DataFrame): Second DataFrame.
            join_column (str): Name of the column to join on.
            join_type (str, optional): Type of join to perform ('left', 'right', 'inner', 'outer'). Default is 'left'.

        Returns:
            DataFrame: Resulting DataFrame from the join.
        """
        return df1.join(df2, on=join_column, how=join_type)

    def correct_col_type(self, dataframe):
        """
        Corrects the data type of columns in a DataFrame.

        Args:
            dataframe (DataFrame): DataFrame to correct.

        Returns:
            DataFrame: DataFrame with corrected column data types.
        """
        for column in dataframe.columns:
            sample_values = dataframe.select(column).na.drop().limit(100).collect()
            sample_values = [row[column] for row in sample_values if row[column] is not None]

            if not sample_values:
                continue

            column_value = sample_values[0]

            if isinstance(column_value, str) and ('.' in column_value or '0.' in column_value or '.0' in column_value):
                dataframe = dataframe.withColumn(column, dataframe[column].cast(FloatType()))
            
            elif isinstance(column_value, str) and column_value.isdigit():
                dataframe = dataframe.withColumn(column, dataframe[column].cast(IntegerType()))
            
            else:
                dataframe = dataframe.withColumn(column, dataframe[column].cast(StringType()))
        
        return dataframe

##################################################################################################################
    def analyze_outliers_oneClass(self, dataframe, target, ax, position, dropOutliers=True):
        """
        Analyzes the outliers in the numerical (non-categorical) columns in a DataFrame using 
        a box plot.
    
        Args:
            dataframe (DataFrame): DataFrame to analyze.
            target (int): Target label for coloring purposes.
            ax (AxesSubplot): Matplotlib Axes object for plotting.
            position (str): Position of the plot (left or right).
            dropOutliers (bool, optional): Whether to drop outliers from the DataFrame. Defaults to True.
        
        Returns:
            DataFrame: Cleaned DataFrame with outliers removed if dropOutliers is True.
        """
        sns.set_style("dark")
        color = 'blue' if target == 0 else 'red'
        numeric_columns = [field.name for field in dataframe.schema.fields if isinstance(field.dataType, (IntegerType, FloatType))]
    
        if not numeric_columns:
            print("No numerical columns found in the DataFrame.")
            return dataframe
    

    
        for i, column in enumerate(numeric_columns):
            data = dataframe.select(column).toPandas()
            sns.boxplot(data=data, ax=ax[i][position], orient='h', color=color)
            
            if dropOutliers:
                Q1 = data[column].quantile(0.25)
                Q3 = data[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                dataframe = dataframe.filter((dataframe[column] >= lower_bound) & (dataframe[column] <= upper_bound))
        return dataframe
    
    def analyze_outliers(self, dataframe0, dataframe1, dropOutliers=True):
        """
        Analyzes the outliers in the numerical (non-categorical) columns in both dataframes associated to each label using 
        a box plot.
    
        Args:
            dataframe0 (pyspark.DataFrame): DataFrame associated to label 0.
            dataframe1 (pyspark.DataFrame): DataFrame associated to label 1.
            dropOutliers (bool, optional): Whether to drop outliers from the DataFrame. Defaults to True.
        
        Returns:
            tuple: Cleaned DataFrames (dataframe0, dataframe1) with outliers removed if dropOutliers is True.
        """
        sns.set_theme()
        numeric_columns = [field.name for field in dataframe0.schema.fields if isinstance(field.dataType, (IntegerType, FloatType))]
        n_plots = len(numeric_columns)
    
        fig, ax = plt.subplots(n_plots, 2, figsize=(16, 1.5 * n_plots))
        
        if n_plots == 1:
            ax = [ax]
    
        cleaned_dataframe0 = self.analyze_outliers_oneClass(dataframe0, target=0, ax=ax, position=0, dropOutliers=dropOutliers)
        cleaned_dataframe1 = self.analyze_outliers_oneClass(dataframe1, target=1, ax=ax, position=1, dropOutliers=dropOutliers)
    
        plt.tight_layout()
        plt.show()
    
        return cleaned_dataframe0, cleaned_dataframe1
    

###############################################################################################################################################


    
    def hist_columns(self, column, data0, data1):
        """
        Generates comparative histograms for a column between two different datasets.

        Args:
            column (str): Name of the column on which the histograms will be generated.
            data0 (DataFrame): DataFrame containing the data for the first group.
            data1 (DataFrame): DataFrame containing the data for the second group.
        """
        sns.set_theme()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))

        sns.histplot(data=data0, x=column, ax=ax1,color='blue', kde=True)
        ax1.set_title("Clients with Payment Difficulties (Label 0)")

        sns.histplot(data=data1, x=column, ax=ax2, color='red', kde=True)
        ax2.set_title("Clients without Payment Difficulties (Label 1)")

        plt.show()

    def count_plot(self, column, data0, data1):
        """
        Generates comparative bar plots for a column between two different datasets.

        Args:
            column (str): Name of the column on which the bar plots will be generated.
            data0 (DataFrame): DataFrame containing the data for the first group.
            data1 (DataFrame): DataFrame containing the data for the second group.
        """
        sns.set_theme()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))

        sns.countplot(data=data0, x=column, ax=ax1, palette='pastel')
        ax1.set_title("Clients with Payment Difficulties (Label 0)")
        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=90)


        sns.countplot(data=data1, x=column, ax=ax2, palette='pastel')
        ax2.set_title("Clients without Payment Difficulties (Label 1)")
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=90)


        plt.show()

    def visualize_data(self, data0, data1):
        """
        Visualizes the data in the DataFrame using appropriate plots for different column types.

        Args:
            dataframe (DataFrame): DataFrame to visualize.
            data0 (DataFrame): DataFrame containing the data for the first group.
            data1 (DataFrame): DataFrame containing the data for the second group.
        """
        data0_pd =  data0.sample(False, 200 / data0.count()).toPandas()
        data1_pd =  data1.sample(False, 200 / data1.count()).toPandas()
        for column in data0.columns:
            # Get the data type of the column
            col_type = [field.dataType for field in data0.schema.fields if field.name == column][0]

            # Check the data type and choose the corresponding plot type
            if isinstance(col_type, FloatType):
                # Histogram for numerical variables
                self.hist_columns(column, data0_pd, data1_pd)
            
            elif isinstance(col_type, (StringType, IntegerType)):
                # Bar plot for categorical or integer variables
                self.count_plot(column, data0_pd, data1_pd)
                   
        
    def analyze_nulls(self, dataframe, table, target, title=""):
        """
        Analyzes the null values in numerical columns of a DataFrame. Displays the percentage of null values with a bar chart.
    
        Args:
            dataframe (DataFrame): DataFrame to analyze.
            table (str): Name of the table (SQL view) on which the queries will be executed.
            target (int): The target variable to decide the color of the bars.

        Returns:
            columns_percentajes_column,null_percentages:  pair of dicionaries with all the null values per column and only columns
                                                            with null values
        """
        color = 'blue' if target == 0 else 'red'
        sns.set_theme()
    
        # Get the schema of the DataFrame to identify numerical columns
        numeric_columns = [field.name for field in dataframe.schema.fields if isinstance(field.dataType, (IntegerType, FloatType))]
        
        
        if not numeric_columns:
            print("No numerical columns found in the DataFrame.")
            return
    
        # Calculate total number of records
        total_records = dataframe.count()
    
        # Calculate percentage of null values for each numerical column
        columns_percentajes_columns={}
        null_percentages = {}
        for column in numeric_columns:
            query = f"SELECT COUNT(*) FROM {table} WHERE {column} IS NULL"
            count_nulls = self.spark.sql(query).collect()[0][0]
            percentage_or = 100*count_nulls / total_records
            columns_percentajes_columns[column]=round(percentage_or,5)
            if percentage_or > 0:  # Only include columns with a non-zero percentage of nulls
                null_percentages[column]=percentage_or
        
        if not null_percentages:
            print("No  columns with null values found in the DataFrame.")
            return
        columns_percentajes_columns['target']=target
        # Extract columns and percentages from the dictionary
        columns, percentages = zip(*null_percentages.items())
    
        # Adjust the width of the figure based on the number of columns
        plt.figure(figsize=(20, 8))
    
        # Create a horizontal bar chart using seaborn
        sns.barplot(x=columns, y=percentages, color=color)
        
        # Rotate y labels for better readability
        plt.xticks(rotation=90, ha='right')
        plt.xlabel('columns')
        plt.ylabel('Percentage of Null Values')
        plt.title(f'{title}')
        plt.show()
        return columns_percentajes_columns, null_percentages


    def _replace_nulls_with_most_common(self, dataframe, table):
        """
        Replaces null values in categorical columns with the most common value in each column using SQL queries.

        Args:
            dataframe (DataFrame): DataFrame to analyze.
            table (str): Name of the table (SQL view) on which the queries will be executed.

        Returns:
            DataFrame: DataFrame with null values replaced by the most common value in each categorical column.
        """
        # Get the schema of the DataFrame to identify categorical columns
        categorical_columns = [field.name for field in dataframe.schema.fields if isinstance(field.dataType, StringType) or isinstance(field.dataType, IntegerType)]

        if not categorical_columns:
            print("No categorical columns found in the DataFrame.")
            return dataframe

        # Iterate through each categorical column
        for column in categorical_columns:
            # Build and execute SQL query to find the most common value
            query = f"SELECT {column}, COUNT(*) AS cnt FROM {table} GROUP BY {column} ORDER BY cnt DESC LIMIT 1"
            most_common_value_row = self.spark.sql(query).collect()[0]

            # Extract the most common value
            most_common_value = most_common_value_row[column]

            # Replace null values with the most common value
            dataframe = dataframe.withColumn(column, when(col(column).isNull(), most_common_value).otherwise(col(column)))

        return dataframe
    
    
    def analysis_nullValues(self, dataframe0, dataframe1, table0, table1, title0, title1, threshold=50, delateNullCols=True, fillNulls_numerical=True, fillNulls_categorical=True):
        """
        Analyzes the null values in numerical columns of two DataFrames and visualizes the results.
    
        This method calculates the percentage of null values for numerical columns in two separate 
        DataFrames. It then merges the results into a single DataFrame, including the absolute 
        difference in the percentage of null values between the two DataFrames. The method also 
        prints the merged DataFrame in a readable format.
    
        Args:
            dataframe0 (DataFrame): The first DataFrame to analyze.
            dataframe1 (DataFrame): The second DataFrame to analyze.
            table0 (str): The name of the table (SQL view) for the first DataFrame.
            table1 (str): The name of the table (SQL view) for the second DataFrame.
            title0 (str): The title for the analysis of the first DataFrame.
            title1 (str): The title for the analysis of the second DataFrame.
            threshold (float, optional): Threshold for determining columns with high null values to drop. Defaults to 0.5.
            delateNullCols (bool, optional): Whether to drop columns that exceed the null value threshold. Defaults to True.
            fillNulls_numerical (bool, optional): Whether to fill null values for numerical columns. Defaults to True.
            fillNulls_categorical (bool, optional): Whether to fill null values for categorical columns. Defaults to True.
            
        Returns:
            DataFrame: The original PySpark DataFrame with columns possibly dropped if delateNullCols is True.
        """
        sns.set_theme()
    
        # Analyze null values on the sample DataFrames
        res1, columnsNull1 = self.analyze_nulls(dataframe1, table1, target=1, title=title1)
        res0, columnsNull0 = self.analyze_nulls(dataframe0, table0, target=0, title=title0)
        df1_nullPerc = pd.DataFrame.from_dict(res1, orient='index')
        df0_nullPerc = pd.DataFrame.from_dict(res0, orient='index')
    
        df1_nullPerc.columns = ['percentage null values label 1']
        df0_nullPerc.columns = ['percentage null values label 0']
    
        merged_df = pd.concat([df1_nullPerc, df0_nullPerc], axis=1)
        merged_df['difference percentages null values'] = np.abs(merged_df['percentage null values label 0'] - merged_df['percentage null values label 1'])
    
        # Show dataframe with the difference in percentages of null values for both datasets
        display(HTML(merged_df.to_html()))
    
        # Drop columns with high null values if delateNullCols is True
        if delateNullCols:
            columns_to_drop = merged_df[merged_df.max(axis=1) > threshold].index.tolist()
            columns_to_drop = [col for col in columns_to_drop if col != 'TARGET']
            dataframe0 = dataframe0.drop(*columns_to_drop)
            dataframe1 = dataframe1.drop(*columns_to_drop)
    
        # Fill null values for numerical values
        if fillNulls_numerical:
            numericalCols = [field.name for field in dataframe0.schema.fields if isinstance(field.dataType, FloatType)]
            
            # Dataframe 0
            mean_values0 = {}
            nullNumericalCols0 = [col for col in numericalCols if col in list(columnsNull0.keys())]
            for column in nullNumericalCols0:
                if columnsNull0[column] > 0:
                    res = dataframe0.select(avg(col(column))).collect()[0]
                    mean_values0[column] = res[0]
            dataframe0 = dataframe0.fillna(mean_values0)
            
            # Dataframe 1
            mean_values1 = {}
            nullNumericalCols1 = [col for col in numericalCols if col in list(columnsNull1.keys())]
            for column in nullNumericalCols1:
                if columnsNull1[column] > 0:
                    res = dataframe1.select(avg(col(column))).collect()[0]
                    mean_values1[column] = res[0]
            dataframe1 = dataframe1.fillna(mean_values1)
            
        # Fill null values for categorical values
        if fillNulls_categorical:
            categorical_columns = [field.name for field in dataframe0.schema.fields if isinstance(field.dataType, StringType) or isinstance(field.dataType, IntegerType)]
            for column in categorical_columns:
                # Dataframe 0
                query = f"SELECT {column}, COUNT(*) AS cnt FROM {table0} GROUP BY {column} ORDER BY cnt DESC LIMIT 1"
                most_common_value_row = self.spark.sql(query).collect()[0]
                most_common_value = most_common_value_row[column]
                dataframe0 = dataframe0.withColumn(column, when(col(column).isNull(), most_common_value).otherwise(col(column)))
    
                # Dataframe 1
                query1 = f"SELECT {column}, COUNT(*) AS cnt FROM {table1} GROUP BY {column} ORDER BY cnt DESC LIMIT 1"
                most_common_value_row1 = self.spark.sql(query1).collect()[0]
                most_common_value1 = most_common_value_row1[column]
                dataframe1 = dataframe1.withColumn(column, when(col(column).isNull(), most_common_value1).otherwise(col(column)))
    
        return dataframe0, dataframe1




    def phik_correlation(self, data, target_column, id_columns, label,color=None,min_unique_values=2, threshold=0.5, dropIrrelevantCols=True, figsize=(20, 20)):
        """Computes and displays the PhiK correlation matrix for categorical columns with respect to the target column.
    
        Args:
            data (pandas.DataFrame): Sample Pandas DataFrame to analyze the PhiK correlation.
            target_column (str): Name of the target column.
            label (int): Label value used to set the color for the heatmap.
            min_unique_values (int, optional): Minimum number of unique values required to include a column. Defaults to 2.
            threshold (float, optional): Threshold for determining highly correlated columns to drop. Defaults to 0.5.
            dropIrrelevantCols (bool, optional): Whether to drop columns that are highly correlated. Defaults to True.
            figsize (tuple, optional): Heatmap correlation matrix size. Defaults to (20, 20).
            
        Returns:
            list: List of columns to be dropped due to high correlation.
        """
        sns.set_style("white")

    
        # Set the color for each label
        if color is None:
            color = "darkred" if label == 0 else "darkblue"
        
        # Obtain the categorical columns 
        categoricalCols = [field.name for field in data.schema.fields if isinstance(field.dataType, StringType)]
        categoricalCols = [col for col in categoricalCols if col not in id_columns]
    
        # Register the DataFrame as a SQL temporary view
        data.createOrReplaceTempView("data_view")
    
        # Construct the SQL query to count distinct values for each categorical column
        count_queries = [f"COUNT(DISTINCT {col}) as {col}" for col in categoricalCols + [target_column]]
        query = f"SELECT {', '.join(count_queries)} FROM data_view"
        
        # Execute the SQL query to get the distinct value counts
        unique_counts = data.sql_ctx.sql(query).collect()[0].asDict()
    
        # Filter out columns with too few unique values
        filtered_cols = [col for col in unique_counts if unique_counts[col] >= min_unique_values]
    
        # Transform the data into a pandas dataframe
        data = data.select(filtered_cols).toPandas()
    
        # Specify categorical and continuous columns explicitly
        categorical_cols = [col for col in filtered_cols if data[col].dtype == 'object' or data[col].nunique() < min_unique_values]
        continuous_cols = [col for col in filtered_cols if col not in categorical_cols]
    
        # Compute PhiK correlation matrix with respect to the target column
        phik_matrix_result = data.phik_matrix(interval_cols=continuous_cols)
        
        to_drop = set()
        retained_cols = set()
    
        # Print and optionally remove highly correlated columns
        if dropIrrelevantCols:
            for i in range(len(phik_matrix_result.columns)):
                for j in range(i + 1, len(phik_matrix_result.columns)):
                    col1 = phik_matrix_result.columns[i]
                    col2 = phik_matrix_result.columns[j]
                    if abs(phik_matrix_result.loc[col1, col2]) >= threshold:
                        if col1 not in to_drop and col1 not in retained_cols:
                            to_drop.add(col2)
                            retained_cols.add(col1)
                        elif col2 not in to_drop and col2 not in retained_cols:
                            to_drop.add(col1)
                            retained_cols.add(col2)
    
        # Create a mask array to hide the upper triangle
        mask_array = np.zeros_like(phik_matrix_result, dtype=bool)
        mask_array[np.triu_indices_from(mask_array)] = True
    
        cmap = LinearSegmentedColormap.from_list("custom_white_red", ["white", color])
        
        # Plot the heatmap
        plt.figure(figsize=figsize)
        sns.heatmap(phik_matrix_result, cmap=cmap, annot=False, mask=mask_array, linewidth=0.2)
        plt.xticks(rotation=90, fontsize=10)
        plt.yticks(rotation=0, fontsize=10)
        plt.title(f"Phi-K Correlation Matrix for categorical columns for dataset labeled by {target_column}")
        plt.show()
        
        return list(to_drop)

    


    def pearson_correlation(self, data, target_column, label,color=None, threshold=0.5, dropIrrelevantCols=True, figsize=(20, 20)):
        """Computes and displays the Pearson correlation matrix for numerical columns with respect to the target column.
        
        Args:
            data (pandas.DataFrame): Sample Pandas DataFrame to analyze the Pearson correlation.
            target_column (str): Name of the target column (variable objetivo).
            label (int): Determines the color scheme based on the label value (0 for darkred, 1 for darkblue).
            threshold (float, optional): Threshold for determining highly correlated columns to drop. Defaults to 0.5.
            dropIrrelevantCols (bool, optional): Whether to drop columns that are highly correlated. Defaults to True.
            figsize (tuple, optional): Heatmap correlation matrix size. Defaults to (20,20).
            
        Returns:
            list: List of columns to be dropped due to high correlation.
        """
        if color is None:
            color = "darkred" if label == 0 else "darkblue"
        sns.set_theme(style="white")
    
        # Concatenate the target column with the data
        numericalCols = [field.name for field in data.schema.fields if isinstance(field.dataType, (FloatType, IntegerType))]
    
        # Compute the correlation matrix
        corr = data.select(numericalCols).toPandas().corr().abs()
    
        to_drop = set()
        retained_cols = set()
    
        # Print and optionally remove highly correlated columns
        if dropIrrelevantCols:
            for i in range(len(corr.columns)):
                for j in range(i + 1, len(corr.columns)):
                    col1 = corr.columns[i]
                    col2 = corr.columns[j]
                    if abs(corr.loc[col1, col2]) >= threshold:
                        if col1 not in to_drop and col1 not in retained_cols:
                            to_drop.add(col2)
                            retained_cols.add(col1)
                        elif col2 not in to_drop and col2 not in retained_cols:
                            to_drop.add(col1)
                            retained_cols.add(col2)
    
        # Generate a mask for the upper triangle
        mask = np.triu(np.ones_like(corr, dtype=bool))
        
        # Set up the matplotlib figure
        f, ax = plt.subplots(figsize=figsize)
        
        # Generate a custom colormap that goes from white to the specified color
        cmap = LinearSegmentedColormap.from_list("custom_white_red", ["white", color])
        
        # Draw the heatmap with the mask and correct aspect ratio
        sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, vmin=0, center=0.5,
                    square=True, linewidths=0.5, cbar_kws={"shrink": .5})
    
        plt.title(f'Correlation matrix (absolute values) for label {label} dataset')
        
        # Show the plot
        plt.show()
        
        return list(to_drop)

    def plot_correlations(self, data, target_column, id_columns, label,color=None, figsize=(20, 20), min_unique_values=2, threshold=0.5):
        """Computes and displays the PhiK and Pearson correlation matrices with respect to the target column.
        
        Args:
            data (pandas.DataFrame): Pandas DataFrame to analyze.
            target_column (str): Name of the target column (variable objetivo).
            id_columns (list): List of columns to exclude from the analysis, typically identifier columns.
            label (int): Label value used to set the color for the heatmap.
            figsize (tuple, optional): Heatmap correlation matrix size. Defaults to (20,20).
            min_unique_values (int, optional): Minimum number of unique values required to include a column. Defaults to 2.
            threshold (float, optional): Threshold for determining highly correlated columns to drop. Defaults to 0.5.
            dropIrrelevantCols (bool, optional): Whether to drop columns that are highly correlated. Defaults to True.
            
        Returns:
            dict: Dictionary with lists of columns to be dropped for both Pearson and PhiK correlations.
        """
        # Call the Pearson correlation function to compute and display the Pearson correlation matrix
        pearson_drops = self.pearson_correlation(data, target_column, label,color, threshold=threshold, figsize=figsize)

        # Call the PhiK correlation function to compute and display the PhiK correlation matrix
        phik_drops = self.phik_correlation(data, target_column, id_columns, label,color, min_unique_values, threshold=threshold, figsize=figsize)

        return {'Pearson': pearson_drops,'PhiK': phik_drops}

        
