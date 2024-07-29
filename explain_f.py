#This code performs the extraction of principal components before and after filtering

# Retain only sensor data
indices = ['sensor measurement' + str(i) for i in GloUse['sensor'][n - 1]]
df1 = df1.loc[:, ['unit', 'cycles'] + indices]

# Normalize the data
for i in indices:
    df1[i] = (df1[i] - df1[i].mean()) / df1[i].std()

# Apply Savitzky-Golay filter to normalized data
data_array = df1.iloc[:, 2:].to_numpy()
smoothed_data_train = np.apply_along_axis(lambda x: savgol_filter(x, window_length=5, polyorder=3), axis=0,
                                          arr=data_array)

# Convert back the smoothed data to DataFrame
df1.iloc[:, 2:] = smoothed_data_train

# Calculate explained variance before and after filtering
pca = PCA()
pca.fit(data_array)
explained_variance_before = pca.explained_variance_ratio_

pca.fit(smoothed_data_train)
explained_variance_after = pca.explained_variance_ratio_

# Create a table for explained variance
ev_table = pd.DataFrame({
    'Principal Component': np.arange(1, len(explained_variance_before) + 1),
    'Explained Variance Before Filtering': explained_variance_before,
    'Explained Variance After Filtering': explained_variance_after
})

