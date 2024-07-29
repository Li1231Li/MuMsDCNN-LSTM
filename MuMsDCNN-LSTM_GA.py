# Build the model
inputs = Input(shape=(30, GloUse['SL'][n - 1], 1))  # Two channels
input1 = layers.Lambda(lambda x: x)(inputs)
input2 = layers.Lambda(lambda x: x)(inputs)

# Define dropout rates
a = 0.6
b = 0.6

# First channel convolution layers
x1 = Conv2D(filters=64, kernel_size=(40, 1), strides=(1, 1), padding='same', activation='tanh')(input1)
x1 = Dropout(a)(x1)
x1 = Conv2D(filters=32, kernel_size=(25, 1), strides=(1, 1), padding='same', activation='tanh')(x1)
x1 = Dropout(a)(x1)
x1 = Conv2D(filters=16, kernel_size=(20, 1), strides=(1, 1), padding='same', activation='tanh')(x1)
x1 = Dropout(a)(x1)
x1 = Conv2D(filters=8, kernel_size=(10, 1), strides=(1, 1), padding='same', activation='tanh')(x1)
x1 = Dropout(a)(x1)
x1 = Conv2D(filters=1, kernel_size=(3, 1), strides=(1, 1), padding='same', activation='tanh')(x1)
x1 = Dropout(a)(x1)
x1 = Flatten()(x1)

# Second channel convolution layers
x2 = Conv2D(filters=128, kernel_size=(10, 1), strides=(1, 1), padding='same', activation='tanh')(input2)
x2 = Dropout(a)(x2)
x2 = Conv2D(filters=64, kernel_size=(10, 1), strides=(1, 1), padding='same', activation='tanh')(x2)
x2 = Dropout(a)(x2)
x2 = Conv2D(filters=32, kernel_size=(10, 1), strides=(1, 1), padding='same', activation='tanh')(x2)
x2 = Dropout(a)(x2)
x2 = Conv2D(filters=32, kernel_size=(10, 1), strides=(1, 1), padding='same', activation='tanh')(x2)
x2 = Dropout(a)(x2)
x2 = Conv2D(filters=1, kernel_size=(3, 1), strides=(1, 1), padding='same', activation='tanh')(x2)
x2 = Dropout(a)(x2)
x2 = Flatten()(x2)

# Concatenate outputs from both channels
x = Concatenate()([x1, x2])

# Add an LSTM layer
x = Reshape((1, -1))(x)  # Flatten features into one dimension
x = LSTM(50, activation='tanh', return_sequences=False)(x)

# Fully connected layers
x = Dropout(b)(x)
x = Dense(100, activation='tanh')(x)
x = Dense(1, name='out')(x)

# Build the model
model = Model(inputs=inputs, outputs=x)

# Compile the model
adam = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(loss='mse', optimizer=adam)

# Add checkpoint callback
checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = ModelCheckpoint(filepath=checkpoint_path,
                              monitor='val_loss',  # Monitor validation loss
                              save_weights_only=True,
                              verbose=1)

# Train the model
history1 = model.fit(train_input, train_output, batch_size=512, epochs=200, shuffle=True)  # Training phase

# Adjust learning rate and recompile model
adam = Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(loss='mse', optimizer=adam)

# Continue training with validation split and callbacks
history2 = model.fit(train_input, train_output, validation_split=0.33, batch_size=512, epochs=50, shuffle=True,
                     callbacks=[cp_callback])  # Testing phase
# Calculate validation loss
val_loss = history2.history['val_loss'][-1]  # Assume the last epoch's validation loss as fitness

# Initialize population
def initialize_population(pop_size):
    population = []
    for _ in range(pop_size):
        learning_rate = random.uniform(0.0001, 0.1)  # Generate random learning rate
        batch_size = random.choice([16, 32, 64, 128])  # Randomly choose from candidate batch sizes
        population.append([learning_rate, batch_size])
    return population

# Genetic algorithm function
def genetic_algorithm(population, pop_size, elite_size, mutation_rate, generations):
    for generation in range(generations):
        # Calculate fitness
        fitness_scores = [fitness_function(ind) for ind in population]

        # Selection
        selected_indices = sorted(range(len(fitness_scores)), key=lambda i: fitness_scores[i])[:elite_size]
        selected_population = [population[i] for i in selected_indices]

        # Crossover and mutation
        children = []
        while len(children) < pop_size - elite_size:
            parent1, parent2 = random.sample(selected_population, 2)
            child = crossover(parent1, parent2)
            child = mutate(child, mutation_rate)
            children.append(child)

        # Update population
        population = selected_population + children

    # Best individual in final population
    best_individual = min(population, key=fitness_function)
    best_learning_rate, best_batch_size = best_individual

    return best_learning_rate, best_batch_size

# Set parameters
pop_size = 20
elite_size = 5
mutation_rate = 0.1
generations = 20

# Initialize population
population = initialize_population(pop_size)

# Run genetic algorithm
best_learning_rate, best_batch_size = genetic_algorithm(population, pop_size, elite_size, mutation_rate, generations)

# Print the best hyperparameters found
print(f"Best learning rate: {best_learning_rate}, Best batch size: {best_batch_size}")
