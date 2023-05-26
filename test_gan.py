import torch
import torch.nn as nn
import torchvision
torch.cuda.device('cuda')
image_size= [1, 28, 28]
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        
        self.model = nn.Sequential(
            nn.Linear(torch.prod(torch.tensor(image_size), dtype=torch.int32), 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, torch.prod(torch.tensor(image_size), dtype=torch.int32)),
            nn.Tanh()
        )

    def forward(self, z):
        #shape of z:batchsize,1 ,28,28
        output = self.model(z) 
        image = output.reshape(z.shape[0], *image_size)
        return image

class Discriminator(nn.Module):

        def __init__(self):
             super(Discriminator, self).__init__()
             self.model = nn.Sequential(
                  nn.Linear(torch.prod(torch.tensor(image_size), dtype=torch.int32), 1024),
                  nn.ReLU(inplace=True),
                  nn.Linear(1024, 512),
                  nn.ReLU(inplace=True),
                  nn.Linear(512, 256),
                  nn.ReLU(inplace=True),
                  nn.Linear(256, 128),
                  nn.ReLU(inplace=True),
                  nn.Linear(128, 1),
                  nn.Sigmoid()
                  )
        def forward(self, image):
        #shape of z:batchsize,1 ,28,28
            prob = self.model(image.reshape(image.shape[0], -1)) 
            return prob
                  
    

#Traning 
dataset = torchvision.datasets.MNIST(root='./mnist_data', train=True, download=True, transform=torchvision.transforms.Compose([
                                        torchvision.transforms.ToTensor(),
                                        torchvision.transforms.Resize(28),
                                        torchvision.transforms.Normalize((0.5,), (0.5,))]) )
batch_size = 32
data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
Generator = Generator().to(device)
Discriminator = Discriminator().to(device)
g_optimizer = torch.optim.Adam(Generator.parameters(), lr=0.0001)
d_optimizer = torch.optim.Adam(Discriminator.parameters(), lr=0.0001)

loss_fn = nn.BCELoss()

num_epochs = 100
latent_dim = 784
for epoch in range(num_epochs):
     for i, mini_batch in enumerate(data_loader):
          gt_images, _ = mini_batch
          gt_images = gt_images.to(device)
          z = torch.randn(batch_size, latent_dim).to(device)       
          pred_images = Generator(z) 
        
          g_optimizer.zero_grad()  
          target = torch.ones(batch_size, 1).to(device)
          g_loss = loss_fn(Discriminator(pred_images), target)
          g_loss.backward()
          g_optimizer.step()


          d_optimizer.zero_grad()
          d_loss = 0.5*(loss_fn(Discriminator(gt_images), target) + loss_fn(Discriminator(pred_images.detach()), torch.zeros(batch_size, 1).to(device)))
          d_loss.backward()
          d_optimizer.step()

          if i % 1000 == 0:
               print('Epoch: %d/%d, Step: %d/%d, D Loss: %f, G Loss: %f' % (epoch, num_epochs, i, len(data_loader), d_loss.item(), g_loss.item()))
               for index, image in enumerate(pred_images):
                    torchvision.utils.save_image(image, f"./generated_images/{epoch}_{index}.png")