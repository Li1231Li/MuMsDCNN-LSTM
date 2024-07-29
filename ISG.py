# Define adaptive filtering function
def adaptive_filtering(data, threshold):
    # Calculate the standard deviation of the data
    data_std = np.std(data)

    # Determine whether to use the Savitzky-Golay (SG) filter based on the threshold
    if data_std < threshold:
        filtered_data = savgol_filter(data, window_length=6, polyorder=5)
    else:
        # Otherwise, use Gaussian filter
        filtered_data = gaussian_filter(data, sigma=2)

    return filtered_data
