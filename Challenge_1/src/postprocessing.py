# Test prediction
test_ids = pd.read_csv(TEST_IDS_CSV)
test_ids['image'] = test_ids['image_id']
test_dataset = SoilDataset(test_ids, TEST_DIR, transform=image_transforms['test'], is_test=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

model.eval()
test_preds = []
image_names = []

with torch.no_grad():
    for images, image_ids in test_loader:
        images = images.to(device)
        outputs = model(images)
        preds = outputs.argmax(1).cpu().numpy()
        test_preds.extend(preds)
        image_names.extend(image_ids)

# Map back to soil type
final_labels = [inv_label_mapping[p] for p in test_preds]
submission = pd.DataFrame({
    'image_id': image_names,
    'soil_type': final_labels
})

submission.to_csv('submission.csv', index=False)
print("submission.csv saved!")
