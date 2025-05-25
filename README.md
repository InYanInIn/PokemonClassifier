# 🐉 Pokémon Type Classifier

A TensorFlow/Keras-based convolutional neural network that categorizes Pokémon images into **Poison**, **Fire**, or **Water** types using data augmentation, training history visualization, and confusion matrix analysis.

## 🚀 Features

- **Selective Filtering**: Trains exclusively on Poison, Fire, and Water types.  
- **Data Augmentation**: Random rotations, shifts, zooms and rescaling to improve generalization.  
- **Custom CNN Architecture**:  
  - Three Conv2D + MaxPooling blocks (32 → 64 → 128 filters)  
  - Dense layer (128 units) with Dropout(0.5)  
  - Softmax output for three classes  
- **Training Pipeline**:  
  - 90% training / 10% validation split  
  - Adam optimizer with categorical cross-entropy loss  
- **Visual Feedback**:  
  - **Accuracy Plot** (`training_history.png`) showing training vs. validation accuracy  
  - **Confusion Matrix** (`confusion_matrix.png`) highlighting per-class performance  
- **Model Persistence**: Saves the final model to `pokemon_model.keras`  

## 🔍 Results

- **High overall accuracy** on the validation set, demonstrating effective augmentation and architecture choice.  
- **Confusion matrix** reveals any remaining misclassifications between closely related types.

## 🛠️ Customization

- **Types**: Modify `SELECTED_TYPES` to include different or additional Pokémon types.  
- **Image Channels**: Switch between `rgb` and `rgba` color modes in `ImageDataGenerator`.  
- **Model Depth**: Adjust the number or size of convolutional layers in `prepare_network()`.  
- **Hyperparameters**: Tweak batch size, learning rate, epochs, or augmentation ranges.

## 📚 References

- [TensorFlow Keras Guide](https://www.tensorflow.org/guide/keras)  
- [scikit-learn ConfusionMatrixDisplay](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.ConfusionMatrixDisplay.html)  

