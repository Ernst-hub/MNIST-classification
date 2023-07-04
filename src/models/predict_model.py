import click
import torch
from model import CNN
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv


@click.command()
@click.argument('model_path', type=click.Path(exists=True))
@click.argument('data_path', type = click.Path())

def main(model_path, data_path):
    logger = logging.getLogger(__name__)
    logger.info('Evaluating model')
    
    # load model
    model = CNN()
    model.load_state_dict(torch.load(model_path + '/model.pt'))
    
    device = torch.device("mps" if torch.backends.mps.is_available() 
                          else "cpu")
    
    model.to(device)
    
    criterion = torch.nn.CrossEntropyLoss()
    
    data = torch.load(data_path + '/testloader.pt')
    
    logger.info(f'Loaded data from {data_path}')
    
    model.eval()
    with torch.inference_mode():
        
        running_loss = 0
        running_accuracy = 0
        for images, labels in data:
            images, labels = images.to(device), labels.to(device)
            _, output = model(images)
            
            loss = criterion(output, labels)
            running_loss += loss.item()
            
            # evaluate accuracy
            top_p, top_class = output.topk(1, dim = 1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy = torch.mean(equals.type(torch.FloatTensor))
            running_accuracy += accuracy
            
        print(f'Loss: {running_loss/len(data)}')
        print(f'Accuracy: {running_accuracy/len(data)}')
        
        
if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
