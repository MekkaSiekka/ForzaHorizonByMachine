from os import getcwd
from keyboard_read.demo import KeyBoardStatus



import time
import cv2
import mss
import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import logging



logging.basicConfig(level=logging.INFO)

def getScreenNumpy():
    with mss.mss() as sct:
        # Part of the screen to capture
        monitor = {"top": 40, "left": 0, "width": 1920, "height": 1080}
        # print(data[0].size())
        scr_img = np.array(sct.grab(monitor))
        cv2.imshow("OpenCV/Numpy normal", scr_img)
        cv2.waitKey(100)
        scr_img = cv2.resize(scr_img, (512,512),interpolation=cv2.INTER_CUBIC)
        scr_img = np.moveaxis(scr_img,-1,0)
        scr_img = scr_img[0:3,:,:] #discard alpha channel
        scr_img = scr_img.astype(float)
        scr_img /= 255
        ll = []
        for i in range(BATCH_SIZE):
            ll.append(scr_img)
        inputs2 = np.stack(ll, axis=0)
        
        # print("after stack shape",inputs2.shape)
        inputs2 = torch.from_numpy(inputs2).float().to(device)
        return inputs2

def getCustomeLabel():
    np_label = np.array([1,1,1,1])
    labels = torch.from_numpy(np_label).float().to(device)
    
    ll = [keyboard_result]
    labels = np.stack(ll,axis = 0)    
    labels = torch.from_numpy(labels).float().to(device) #todo: fix this !!
    print(labels)
    #print(labels)
    return labels 



# functions to show an image


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()



import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, 3, padding = 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(8, 16, 3, padding = 1)
        self.fc1 = nn.Linear(16 * 128 * 128, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 4)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        #print(x.shape)
        x = x.reshape(-1, 16 * 128 * 128)  #what is the progblem if use view?
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


if __name__ == "__main__":
    logging.info("logging started")
    
    BATCH_SIZE = 1
    
    keyboard_result =  np.array([0,0,0,0])
    kb_status = KeyBoardStatus(keyboard_result)
    kb_status.start()
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    '''
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                            shuffle=True, num_workers=0)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                            shuffle=False, num_workers=0)

    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    
    # get some random training images
    dataiter = iter(trainloader)
    images, labels = dataiter.next()

    # show images
    #imshow(torchvision.utils.make_grid(images))
    # print labels
    print(' '.join('%5s' % classes[labels[j]] for j in range(4)))
    '''

    net = Net()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    # Assuming that we are on a CUDA machine, this should print a CUDA device:

    logging.info("The device is selected as %s" % device)
    net.to(device)


    import torch.optim as optim

    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


    for epoch in range(2):
        running_loss = 0.0
        i = 0
        while 1:
            i += 1
            # get the inputs; data is a list of [inputs, labels]
            #inputs, labels = data
            # inputs, labels = data[0].to(device), data[1].to(device)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            myinputs = getScreenNumpy()
            outputs = net(myinputs)
            print("outputs", outputs)
            labels = getCustomeLabel()
            #print(labels)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            logging.info('instant loss %f',loss.item())
            if i % 200 == 0:    # print every 2000 mini-batches
                logging.info('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 200))
                running_loss = 0.0

    logging.info('Finished Training')