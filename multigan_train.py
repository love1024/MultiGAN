import queue
import threading
import os
import subprocess
import torch
from shutil import copyfile, rmtree

# Base directory for training_data
BASE_DIR = './training_data/'
DIST_DIR = './pretrained/training_data/'


# Get number of GPUs available
NUM_GPUS = torch.cuda.device_count()

def initiate_queue(queue, gpu_id):
    """
    Take images from queue and start training_data on the gpu
    given to the current thread
    Args:
        queue: Queue from where this thread will take images for training_data
        gpu_id: GPU available to this thread for training_data

    Returns: None
    """

    # Listen while queue is not empty
    while True:
        try:
            filename = queue.get()
        except queue.Empty:
            return

        # Using InGAN, learn distribution for the current image
        output_path = './pretrained'
        cmd = [
            'python', '-m InGAN_lib.train.py',
            '--input_image_path', filename,
            '--output_dir_path','./pretrained',
            '--gpu_id', str(gpu_id),
            '--name', filename,
            '--create_code_copy', 'False']
        command = " ".join(cmd)
        subprocess.call(command, env=os.environ.copy())

        # Mark it as done
        queue.task_done()

def store_distributions():
    """
    Store pretrained in the distribution directory after training_data
    """
    for dist in os.listdir(DIST_DIR):
        name = dist.split('_')[0].split('.')[0]
        source = f'./pretrained/training_data/{dist}/checkpoint_0055000.pth.tar'
        destination = f'./pretrained/{name}.tar'
        copyfile(source, destination)

def clear_data():
    """
    Clear temporary files after learning pretrained
    """
    rmtree('./pretrained/training_data', ignore_errors=True)

def start_training():

    # If there is no GPU available, stop training_data
    if (NUM_GPUS == 0):
        print("No GPU avaiable for training_data")
        return

    # Define a queue to store all images
    q = queue.Queue()

    # Read all images and put in queue
    for imgname in os.listdir(BASE_DIR):
        q.put_nowait(BASE_DIR + imgname)

    # Start threads equal to the number of available GPUs
    for gpu_id in range(NUM_GPUS):
        thread = threading.Thread(target=initiate_queue, args=(q, gpu_id))
        thread.setDaemon(True)
        thread.start()

    # Wait until all threads are not completed
    q.join()

    # Store all found pretrained in a single directory
    store_distributions()

    # Clear all temporary data
    clear_data()

if __name__ == '__main__':
    start_training()

