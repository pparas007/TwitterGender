import pandas as pd

#produces descriptive stats about a pandas.core.series.Series
#flag = 0, categorical series, by element of a Series
#flag = 1, categorical series, by element length of a Series
#flag = 2, categorical series, no extra info
def descriptive_stats(series, file, flag = 0):
    #<class 'pandas.core.frame.DataFrame'>
    #<class 'pandas.core.series.Series'>
    #print(type(series))
    #if type(series) == 'pandas.core.series.Series':
    #    print("Series")
    
    file.write("\n")
    file.write("Stats for series: " + series.name + "\n")
    file.write("-------------------------------------------\n")
    file.write("Length:" + str(len(series)) + "\n")
    file.write("# of NaN or Nulls " + str(series.isna().sum()) + "\n")
    file.write("Describe output:" + "\n")
    file.write(str(series.describe(include = 'all')) + "\n")
    file.write("categorical:" + str(pd.api.types.is_categorical_dtype(series)) + "\n")
    if pd.api.types.is_categorical_dtype(series):
        if flag == 1:
            seriesDF = series.str.len().value_counts().to_frame(name = 'count')
            seriesDF.rename_axis("color_len", inplace = True)
            total = seriesDF ['count'].sum()
            seriesDF ['perc'] = seriesDF['count'] * 100 / total
            file.write(str(seriesDF) + "\n")
        elif flag == 0:
            file.write("Unique values:\n")
            file.write(str(series.unique()) + "\n")
            file.write("Value counts output:\n")
            file.write(str(series.value_counts(dropna = False)) + "\n")
    else:
        file.write("Value counts output in bins:\n")
        file.write(str(series.value_counts(bins = 10)) + "\n")
    file.write("\n")

#6:gender_confidence, 8:confidence in profile, 10:description, 11:no of favourited tweets,
#13:link color, 17:retweet count, 18:sidebar color, 19:tweet text, 21:tweet count
def stats(X, y):
    #creates stats_dataset.txt file with stats of the dataset
    file = open("stats_dataset.txt", "w")
    descriptive_stats(y.astype('category'), file)
    descriptive_stats(X.iloc[:, 0], file)
    descriptive_stats(X.iloc[:, 1], file)
    descriptive_stats(X.iloc[:, 2].astype('category'), file, 2)
    descriptive_stats(X.iloc[:, 3], file)
    descriptive_stats(X.iloc[:, 4].astype('category'), file, 1)
    descriptive_stats(X.iloc[:, 5], file)
    descriptive_stats(X.iloc[:, 6].astype('category'), file, 1)
    descriptive_stats(X.iloc[:, 7].astype('category'), file, 2)
    descriptive_stats(X.iloc[:, 8], file)
    file.close()