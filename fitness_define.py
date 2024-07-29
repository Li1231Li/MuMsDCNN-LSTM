#This code defines fitness_function, which allows switching between 4 sets of data by modifying the value of n.

def fitness_function(params):
    learning_rate = params[0]
    batch_size = params[1]
    if __name__ == '__main__':
        n = 1
    # Building training data..
    df = pd.read_pickle('Data/' + GloUse['train_file'][n - 1] + '.pickle')  # Post-filtered (previously changed)
    # df = pd.read_pickle('Data/' + GloUse['train_file'][n - 1] + '.pickle')  # Pre-filtered
    train_input = df.iloc[:, :-2].values.reshape(-1, 30, GloUse['SL'][n - 1], 1)
    train_output = df.iloc[:, -1].values.reshape(-1, )
    # Building test data
    df = pd.read_pickle('Data/' + GloUse['test_file'][n - 1] + '.pickle')
    df1 = []
    df2 = []
    for i in range(GloUse['test_units'][n - 1]):
        if (i + 1) in (df.unit.values):
            df1.append(df[df.unit == i + 1].iloc[-1, :-2].values)
            df2.append(df[df.unit == i + 1].iloc[-1, -1])
    test_input = np.array(df1).reshape(-1, 30, GloUse['SL'][n - 1], 1)
    test_output = np.array(df2).reshape(-1, )
