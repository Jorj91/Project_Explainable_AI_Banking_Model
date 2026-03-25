import torch
import torch.nn as nn
import os

## Setup functions

'''# Training Setup
criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# optimizer with weight decay (to add L2 regularization)
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-4,
    weight_decay=1e-4
)
# Move model to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)'''


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_criterion():
    return nn.CrossEntropyLoss()


def get_optimizer(model, lr=1e-4, weight_decay=1e-4):
    return torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )


## Training Loop

# Training Loop
# An epoch = one full pass through the training dataset

'''for epoch in range(10):
    # Set the model to training mode to enable dropout and batch normalization updates
    model.train()
    # variable to accumulate the loss over the epoch
    running_loss = 0

    # variable to accumulate the loss over the epoch
    for images, labels in train_loader:
        # move the batch to the selected device
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass:
        # The images are passed through the network to produce predictions
        outputs = model(images)
        # Compute the loss (CrossEntropyLoss) between predictions and ground truth labels
        loss = criterion(outputs, labels)

        # Reset gradients from the previous iteration
        optimizer.zero_grad()

        # Backpropagation: Compute gradients of the loss with respect to model parameters
        loss.backward()
        # Update model weights using the optimizer
        optimizer.step()
        # Accumulate the batch loss for reporting
        running_loss += loss.item()

    # Print the average loss for the epoch
    print("Epoch:", epoch+1, "Loss:", running_loss/len(train_loader))'''


def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    epochs=10,
    patience=3,
    save_path="results/models/best_model.pth"
):
    best_val_acc = 0
    patience_counter = 0

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    for epoch in range(epochs):

        # ===== TRAIN =====
        model.train()
        running_loss = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)

        # ===== VALIDATION =====
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        print(f"\nEpoch {epoch+1}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        # ===== SAVE BEST MODEL =====
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            

            torch.save(
                {"model_state": model.state_dict(),
                 "best_val_acc": best_val_acc,
                 "epoch": epoch
                 },
                save_path
            )

            print(f"✅ Best model saved! (Val Acc: {best_val_acc:.4f})")

        else:
            patience_counter += 1
            print(f"⏳ No improvement ({patience_counter}/{patience})")

        # ===== EARLY STOPPING =====
        if patience_counter >= patience:
            print("🛑 Early stopping triggered")
            break

    print(f"\nTraining completed. Best Val Acc: {best_val_acc:.4f}")


## Evaluation

'''# After training, we evaluate the model on the validation dataset

# Switch the model to evaluation mode
model.eval()

correct = 0 # number of correctly classified samples
total = 0 # total number of samples evaluated

# Disable gradient computation to speed up evaluation and reduce memory usage
with torch.no_grad():

    # Iterate through the validation dataset
    for images, labels in val_loader:

        images = images.to(device)
        labels = labels.to(device)
        # Forward pass
        outputs = model(images)

        # Get the predicted class index
        # torch.max returns (value, index)
        _, predicted = torch.max(outputs,1)
        # Update total number of samples
        total += labels.size(0)
        # Count how many predictions are correct
        correct += (predicted == labels).sum().item()

print("Validation Accuracy:", correct/total)


# Switch the model to evaluation mode
model.eval()

correct = 0 # number of correctly classified samples
total = 0 # total number of samples evaluated

# Disable gradient computation to speed up evaluation and reduce memory usage
with torch.no_grad():

    # Iterate through the test dataset
    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)
        # Forward pass
        outputs = model(images)

        # Get the predicted class index
        # torch.max returns (value, index)
        _, predicted = torch.max(outputs,1)
        # Update total number of samples
        total += labels.size(0)
        # Count how many predictions are correct
        correct += (predicted == labels).sum().item()

print("Test Accuracy:", correct/total)'''


# Evaluation (used for both val + test)
def evaluate(model, loader, criterion, device):
    model.eval()

    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    return total_loss / len(loader), correct / total



'''def save_model(model, path):
    torch.save({"model_state": model.state_dict()}, path)'''


def load_model(model, path, device):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    return model