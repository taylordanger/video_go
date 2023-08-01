import os
import cv2
import time
import torch
from PIL import Image
from torch import nn, optim, ones, zeros, randn
from torchvision import transforms, utils
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader

# Hyperparameters
z_dim = 100
learning_rate = 0.0002
batch_size = 64
num_epochs = 5

# Check if CUDA is available and set device to GPU if it is
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
dataset = CIFAR10(root='./data', download=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

class Generator(nn.Module):
    def __init__(self, input_dim=100, output_dim=3, input_size=256):
        super(Generator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_size = input_size

        self.fc = nn.Sequential(
            nn.Linear(self.input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 128 * (self.input_size // 4) * (self.input_size // 4)),
            nn.BatchNorm1d(128 * (self.input_size // 4) * (self.input_size // 4)),
            nn.ReLU(),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, self.output_dim, 4, 2, 1),
            nn.Tanh(),
        )
        # utils.initialize_weights(self)

    def forward(self, input):
        x = self.fc(input)
        x = x.view(-1, 128, (self.input_size // 4), (self.input_size // 4))
        x = self.deconv(x)

        return x

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        def discriminator_block(in_filters, out_filters, normalization=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(3, 64, normalization=False),  # Adjust the number of input channels to match your images
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, img):
        return self.model(img)

# Create the networks
generator = Generator(input_dim=z_dim, output_dim=3, input_size=256).to(device)
discriminator = Discriminator().to(device)

# Create the optimizers
g_optimizer = optim.Adam(generator.parameters(), lr=learning_rate)
d_optimizer = optim.Adam(discriminator.parameters(), lr=learning_rate)

# Use binary cross entropy loss
criterion = nn.BCELoss()

def noise(size):
    return randn(size, z_dim).to(device)

for epoch in range(num_epochs):
    for n_batch, (real_batch,_) in enumerate(dataloader):
        real_batch = real_batch.to(device)
        batch_size = real_batch.size(0)

         # Train discriminator
        d_optimizer.zero_grad()
        real_labels = ones(batch_size).to(device)
        fake_labels = zeros(batch_size).to(device)
        output = discriminator(real_batch)
        d_real_loss = criterion(output, real_labels)

        noise_vectors = noise(batch_size)
        fake_images = generator(noise_vectors)
        output = discriminator(fake_images.detach())
        d_fake_loss = criterion(output, fake_labels)

        d_loss = d_real_loss + d_fake_loss
        d_loss.backward()
        d_optimizer.step()

        # Train generator
        g_optimizer.zero_grad()
        output = discriminator(fake_images).view(-1)
        g_loss = criterion(output, real_labels)
        g_loss.backward()
        g_optimizer.step()

    print(f"Epoch [{epoch + 1}/{num_epochs}] d_loss: {d_loss.item()} g_loss: {g_loss.item()}")

def video_to_frames(video, path_output_dir):
    """Function to split video into frames."""

    # Attempt to open the video file.
    try:
        vidcap = cv2.VideoCapture(video)
    except Exception as e:
        print(f"Failed to open video file {video}. Error: {e}")
        return

    # If the video file could not be opened, print an error and return.
    if not vidcap.isOpened():
        print(f"Could not open video file {video}.")
        return

    # Attempt to create the output directory.
    try:
        os.makedirs(path_output_dir, exist_ok=True)
    except Exception as e:
        print(f"Failed to create output directory {path_output_dir}. Error: {e}")
        return

    count = 0
    while vidcap.isOpened():
        try:
            success, image = vidcap.read()
        except Exception as e:
            print(f"Failed to read frame from video file {video}. Error: {e}")
            break

        if success:
            try:
                cv2.imwrite(os.path.join(path_output_dir, '%d.png') % count, image)
            except Exception as e:
                print(f"Failed to write frame to file. Error: {e}")
                break
            count += 1
        else:
            break

    # Release the video file and destroy all windows.
    vidcap.release()
    cv2.destroyAllWindows() 

def images_to_vectors(images):
    return images.view(images.size(0), 3072)

def vectors_to_images(vectors):
    return vectors.view(vectors.size(0), 3, 32, 32)

def transform_images(image_directory):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    image_paths = [os.path.join(image_directory, image) for image in os.listdir(image_directory) if image.endswith(".png")]
    images = [transform(Image.open(image_path)) for image_path in image_paths]
    return torch.stack(images)

def frames_to_video(inputpath, outputpath, fps):
    """Function to combine frames into a video."""

    image_array = []

    try:
        files = [f for f in os.listdir(inputpath) if isfile(join(inputpath, f))]
    except Exception as e:
        print(f"Failed to list files in directory {inputpath}. Error: {e}")
        return

    files.sort(key=lambda x: int(x[5:-4]))

    for i in range(len(files)):
        try:
            img = cv2.imread(inputpath + files[i])
            size = (img.shape[1], img.shape[0])
            img = cv2.resize(img, size)
        except Exception as e:
            print(f"Failed to read and resize image file {files[i]}. Error: {e}")
            continue

        image_array.append(img)

    try:
        fourcc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')
        out = cv2.VideoWriter(outputpath, fourcc, fps, size)
    except Exception as e:
        print(f"Failed to create VideoWriter. Error: {e}")
        return

    for i in range(len(image_array)):
        try:
            out.write(image_array[i])
        except Exception as e:
            print(f"Failed to write frame to video. Error: {e}")
            break

    out.release()
# where the stuff is at
video_directory = '/home/darkstar/Desktop/videos/'
base_directory = '/home/darkstar/Desktop/output/'
output_directory = '/home/darkstar/Desktop/output_images/'

try:
    video_paths = [os.path.join(video_directory, video) for video in os.listdir(video_directory) if video.endswith(".mp4")]
except Exception as e:
    print(f"Failed to list mp4 files in directory {video_directory}. Error: {e}")
    video_paths = []

loss = nn.BCELoss()

# Optimizers
g_optimizer = optim.Adam(generator.parameters(), lr=0.0002)
d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002)

def ones_target(size):
    '''
    Tensor containing ones, with shape = size
    '''
    data = torch.ones(size, 1)
    return data

def zeros_target(size):
    '''
    Tensor containing zeros, with shape = size
    '''
    data = torch.zeros(size, 1)
    return data

def train_discriminator(optimizer, real_data, fake_data):
    N = real_data.size(0)
    # Reset gradients
    optimizer.zero_grad()

    # 1.1 Train on Real Data
    prediction_real = discriminator(real_data)
    # Calculate error and backpropagate
    error_real = loss(prediction_real, ones_target(N))
    error_real.backward()

    # 1.2 Train on Fake Data
    prediction_fake = discriminator(fake_data)
    # Calculate error and backpropagate
    error_fake = loss(prediction_fake, zeros_target(N))
    error_fake.backward()

    # 1.3 Update weights with gradients
    optimizer.step()

    # Return error and predictions for real and fake inputs
    return error_real + error_fake, prediction_real, prediction_fake

def train_generator(optimizer, fake_data):
    N = fake_data.size(0)
    # Reset gradients
    optimizer.zero_grad()
    # Sample noise and generate fake data
    prediction = discriminator(fake_data)
    # Calculate error and backpropagate
    error = loss(prediction, ones_target(N))
    error.backward()
