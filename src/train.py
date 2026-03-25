import torch
import torch.nn as nn
import os

## Setup functions

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
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # accuracy tracking
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / len(train_loader)
        train_acc = correct / total

        # ===== VALIDATION =====
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        print(f"\nEpoch {epoch+1}")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        # ===== SAVE BEST MODEL =====
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            

            torch.save(
                {"model_state": model.state_dict(),
                 "best_val_acc": best_val_acc,
                 "epoch": epoch + 1
                 },
                save_path
            )

        else:
            patience_counter += 1
            print(f"⏳ No improvement ({patience_counter}/{patience})")

        # ===== EARLY STOPPING =====
        if patience_counter >= patience:
            print("🛑 Early stopping triggered")
            break

    print(f"\nTraining completed. \nBest Validation Accuracy: {best_val_acc:.4f}")
    print(f"\n✅ Best model saved at: {save_path}")


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



def load_model(model, path, device):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    return model
