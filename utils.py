# import seaborn as sns
# import numpy as np
#
# cf_matrix = [[86, 9, 5],
#              [7, 80, 13],
#              [10, 4, 86]]
#
# sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=True,
#             fmt='.2%', cmap='Blues')

# 2nd Version

# import seaborn as sns
# import matplotlib.pyplot as plt
#
# # Define your confusion matrix data
# cf_matrix = [
#     [86, 9, 5],
#     [7, 79, 13],
#     [10, 8, 82]
# ]
# class_labels = ['Bacterial Red Disease', 'Bacterial Gill Disease', 'Healthy Fish']
# # Plot the confusion matrix using Seaborn's heatmap
# sns.heatmap(cf_matrix, annot=True, fmt='d', cmap='Blues')
#
# # Add labels and title
# plt.xlabel('Predicted Class')
# plt.ylabel('Actual Class')
# plt.title('Confusion Matrix')
#
# # Set axis labels
# plt.xticks(ticks=[0.5, 1.5, 2.5], labels=class_labels)
# plt.yticks(ticks=[0.5, 1.5, 2.5], labels=class_labels)
#
# # Show plot
# plt.show()

import seaborn as sns
import matplotlib.pyplot as plt

# Define your confusion matrix data
# cf_matrix = [
#     [86, 9, 5],
#     [7, 79, 13],
#     [10, 8, 82]
# ]
# class_labels = ['Bacterial Red Disease', 'Bacterial Gill Disease', 'Healthy Fish']
#
# # Plot the confusion matrix using Seaborn's heatmap
# sns.heatmap(cf_matrix, annot=True, fmt='d', cmap = 'Blues')
#
# # Add labels and title
# plt.xlabel('Predicted Class')
# plt.ylabel('Actual Class')
# plt.title('Confusion Matrix')
#
# # Set axis labels with rotated ticks
# plt.xticks(ticks=[0.5, 1.5, 2.5], labels=class_labels, rotation=0)
# plt.yticks(ticks=[0.5, 1.5, 2.5], labels=class_labels, rotation=0)
#
# # Show plot
# plt.show()



#Accuracy vs Epoaxh

import numpy as np
import matplotlib.pyplot as plt

# Given confusion matrix (repeated for demonstration purposes)
confusion_matrices = [
    [[86, 9, 5], [7, 79, 13], [10, 8, 82]],  # Example confusion matrix for epoch 1
    [[88, 7, 5], [6, 80, 14], [9, 7, 84]],   # Example confusion matrix for epoch 2
    [[87, 8, 5], [5, 81, 14], [8, 6, 86]],   # Example confusion matrix for epoch 3
    # Add more confusion matrices for more epochs as needed
]

# Calculate accuracy for each epoch
accuracies = []
for cm in confusion_matrices:
    cm = np.array(cm)
    accuracy = np.trace(cm) / np.sum(cm)
    accuracies.append(accuracy)

# Generate the epochs range
epochs = range(1, len(accuracies) + 1)

# Plot Accuracy vs. Number of Epochs
plt.plot(epochs, accuracies, marker='o')
plt.xlabel('Number of Epochs')
plt.ylabel('Accuracy')
plt.title('Accuracy vs. Number of Epochs')
plt.ylim([0, 1])  # Accuracy ranges from 0 to 1
plt.grid(True)
plt.show()
