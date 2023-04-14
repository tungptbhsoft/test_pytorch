import config
from dataset import SiameseDataset
from torch.utils.data import DataLoader, Dataset
from model import SiameseNetwork
from contrastive import ContrastiveLoss
from torch import optim
import torchvision.transforms as transforms
import torch

# load the dataset
training_dir = config.training_dir
testing_dir = config.testing_dir
training_csv = config.training_csv
testing_csv = config.testing_csv

# Load the the dataset from raw image folders
siaTrans = transforms.Compose([transforms.Resize((105,105)), transforms.ToTensor()])
siamese_dataset = SiameseDataset(training_csv = training_csv,
                                 training_dir = training_dir,
                                 transform = siaTrans)

# Load the dataset as pytorch tensors using dataloader
train_dataloader = DataLoader(siamese_dataset, shuffle=True, num_workers=8, batch_size=8)

# Declare Siamese Network
net = SiameseNetwork().cuda()
criterion = ContrastiveLoss()
optimizer = optim.RMSprop(net.parameters(), lr=1e-4, alpha=0.99, eps=1e-8, weight_decay=0.0005, momentum=0.9)

def train():
    counter = []
    loss_history = [] 
    iteration_number= 0
    epoch = 20
    
    for epoch in range(0, epoch):
        for i, data in enumerate(train_dataloader,0):
            img0, img1 , label = data
            img0, img1 , label = img0.cuda(), img1.cuda() , label.cuda()
            optimizer.zero_grad()
            output1,output2 = net(img0, img1)
            loss_contrastive = criterion(output1,output2,label)
            loss_contrastive.backward()
            optimizer.step()
            if i %50 == 0 :
                print("Epoch number {}\n Current loss {}\n".format(epoch,loss_contrastive.item()))
                iteration_number +=10
                counter.append(iteration_number)
                loss_history.append(loss_contrastive.item())
    return net

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Train the model
model = train()
torch.save(model.state_dict(), "/content/model.pt")
print("Model Saved Successfully")
     

# Load the saved model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SiameseNetwork().to(device)
model.load_state_dict(torch.load("/content/model.pt"))
     

# Load the test dataset
siaTransTest=transforms.Compose([transforms.Resize((105,105)), transforms.ToTensor()])
test_dataset = SiameseDataset(training_csv = testing_csv,
                              training_dir = testing_dir,
                              transform = siaTransTest)

test_dataloader = DataLoader(test_dataset, num_workers=6, batch_size=1, shuffle=True)