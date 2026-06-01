import matplotlib.pyplot as plt

# Sample feature importance values
features = ['Word Count', 'Positive Words', 'Negative Words', 'Sentence Length']
importance = [0.35, 0.40, 0.15, 0.10]

# Plot graph
plt.bar(features, importance)
plt.title("Feature Importance")
plt.xlabel("Features")
plt.ylabel("Importance Score")

# Save graph
plt.savefig("feature_importance.png")

# Show graph
plt.show()

print("Responsible AI analysis completed.")