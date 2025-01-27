import matplotlib.pyplot as plt

# Data for the heart dataset
max_depths = [0, 1, 2, 4]
train_errors = [0.4900, 0.2150, 0.2150, 0.1250]
test_errors = [0.4021, 0.2784, 0.2784, 0.2371]

# Create the plot
plt.figure(figsize=(8, 6))
plt.plot(max_depths, train_errors, label='Training Error', marker='o')
plt.plot(max_depths, test_errors, label='Testing Error', marker='o')

# Adding labels and title
plt.xlabel('Max Depth')
plt.ylabel('Error')
plt.title('Training and Testing Errors vs Max Depth (Heart Dataset)')
plt.legend()

# Save the plot as a PNG file
plt.savefig('heart.png')

# Show the plot
plt.show()
