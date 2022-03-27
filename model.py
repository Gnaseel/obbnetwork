import torch
class Mini_model(torch.nn.Module):
    def __init__(self):
        super(Mini_model, self).__init__()
        #------------------------------- ENCODER ------------------------------------------
        self.encoder1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, kernel_size=3, stride = 1, padding = 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride = 2),
            torch.nn.Conv2d(32, 64, kernel_size=3, stride = 1, padding = 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride = 2)
        )
        self.encoder2 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, kernel_size=3, stride = 1, padding = 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride = 2),
            torch.nn.Flatten()
        )
        self.fc1 = torch.nn.Linear(128*15*15, 100)
        self.fc2 = torch.nn.Linear(100, 8)

    def forward(self, x):
        out = self.encoder1(x)
        out = self.encoder2(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out
