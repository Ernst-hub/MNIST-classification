# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import torch
from torchvision import datasets, transforms
from torch import nn
from torch import optim
from model import CNN
import matplotlib.pyplot as plt
from helper import save_loss_plot
import os
import datetime

@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
@click.argument('figure_filepath', type=click.Path())
@click.argument('num_epochs', type=int, default=10)

def main(input_filepath, output_filepath, figure_filepath, num_epochs):
    """ Trains the model
        saves model in output filepath.
    """
    logger = logging.getLogger(__name__)
    logger.info('Training model')
    
    # setup model and specifics
    model = CNN()
    device = torch.device("mps" if torch.backends.mps.is_available() 
                          else "cpu")

    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # move
    trainloader = torch.load(input_filepath + "/trainloader.pt")
    testloader = torch.load(input_filepath + "/testloader.pt")
    
    # create output directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    subfolder_name = f"model_{timestamp}"
    subfolder_path = os.path.join(output_filepath, subfolder_name)
    os.makedirs(subfolder_path, exist_ok = True)
    output_filepath = subfolder_path
    
    # train
    
    train_loss_log = []
    train_accuracy_log = []
    test_loss_log =[]
    test_accuracy_log = []
    test_acc = 0
    
    for epoch in range(num_epochs):
        running_loss = 0
        running_accuracy = 0
        test_loss = 0
        test_accuracy = 0 
        
        model.train()
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            _, ps = model(images)
            loss = criterion(ps, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            # calculate accuracy
            top_p, top_class = ps.topk(1, dim = 1)
            equals = top_class == labels.view(*top_class.shape)
            running_accuracy += torch.mean(equals.type(torch.FloatTensor))
            
        else:
            print(f"Training loss: {running_loss/len(trainloader)}")
            print(f"Training accuracy: {running_accuracy/len(trainloader)}")
            train_loss_log.append([epoch, running_loss/len(trainloader)])
            train_accuracy_log.append([epoch, running_accuracy/len(trainloader)])
            
            # test
            model.eval()
            with torch.inference_mode():
                
                
                for images, labels in testloader:
                    images, labels = images.to(device), labels.to(device)
                    
                    _, ps = model(images)
                    loss = criterion(ps, labels)
                    test_loss += loss.item()
                    
                    # calculate accuracy
                    top_p, top_class = ps.topk(1, dim = 1)
                    equals = top_class == labels.view(*top_class.shape)
                    test_accuracy += torch.mean(equals.type(torch.FloatTensor))
                    
                else:
                    print(f"Test loss: {test_loss/len(testloader)}")
                    print(f"Test accuracy: {test_accuracy/len(testloader)}")
                    
                    test_loss_log.append([epoch, test_loss/len(testloader)])
                    test_accuracy_log.append([epoch, test_accuracy/len(testloader)])
        
                    if test_accuracy/len(testloader) > test_acc:
                        torch.save(model.state_dict(), output_filepath + "/model.pt")
                        print("Model saved")
                        test_acc = test_accuracy/len(testloader)
        
        

    # save plot loss curve
    save_loss_plot(train_loss_log, figure_filepath + "/train_loss_curve.png")
    save_loss_plot(train_accuracy_log, figure_filepath + "/train_accuracy_curve.png")
    save_loss_plot(test_loss_log, figure_filepath + "/test_loss_curve.png")
    save_loss_plot(test_accuracy_log, figure_filepath + "/test_accuracy_curve.png")
          
        
        
        
    
    
    
    
    

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
