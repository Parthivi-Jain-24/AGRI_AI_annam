# Predict on test set
# Output: 1 = inlier (soil), -1 = outlier (non-soil)
svm_preds = svm.predict(test_features)
binary_preds = [1 if p == 1 else 0 for p in svm_preds]  # Convert to 1/0

# Save Submission
submission = pd.DataFrame({
    'image_id': test_ids,
    'label': binary_preds
})
submission.to_csv('submission.csv', index=False)
print(" Submission file saved as 'submission.csv'")
