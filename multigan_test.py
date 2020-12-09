from InGAN_lib import test
from InGAN_lib.configs import Config;
from torchvision import models
import torchvision.transforms as transforms
import torch
from torch.autograd import Variable
import torch.nn as nn
from os import listdir
from os.path import isfile, join
from multiprocessing import Pool
from itertools import product
from InGAN_lib.InGAN import InGAN
from InGAN_lib import util


DIST_PATH = './pretrained/'
TEST_PATH = './test_image/'
RESULT_PATH = './test_results/'


def read_data(path, divide_by):
    """
    Read data from given directory
    path: Path of data
    divide_by: keep dimensions of image multiple of divide_by
    """
    input_images = [util.read_shave_tensorize(path, divide_by)]
    return input_images

def get_feature_vector(img, model, layer):
    """
    Get feature vector for the given image
    img: Image name
    model: Model to use for feature extraction
    layer: Layer from which to take feature map
    """
    # Normalize image first
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    # Transform image to tensor
    img_tensor = transforms.ToTensor()

    # Create a PyTorch variable with the transformed image
    t_img = Variable(normalize(img_tensor(img)).unsqueeze(0))

    # Create a vector of zeros that will hold our feature vector
    # The lst layer has an output size of 2048
    output = torch.zeros(2048)

    # Define a function that will copy the output of a layer
    def copy_data(m, i, o):
        output.copy_(o.data.reshape(o.data.size(1)))

    # Attach that function to our selected layer
    h = layer.register_forward_hook(copy_data)

    # Run the model on our transformed image
    model(t_img)

    # Detach our copy function from the layer
    h.remove()

    return output

def initiate_parallel_network(distributions, input_tensor):
    """
    Initiate parallel siamese neural network(only works on linux)
    Run multiple instances equal to the number of distributions
    distributions: All distributions
    input_tensor: Input image tensor
    """
    in_size = input_tensor.shape[2:]
    out_size = (in_size[0], in_size[0])

    # Read configuration
    conf = Config().parse(create_dir_flag=False)

    # Load pretrained Resnet50 model for extracting features
    model = models.resnet50(pretrained=True)

    # Get the layer from where we will get features
    layer = model._modules.get('avgpool')
    model.eval()

    max_match = 0
    max_match_idx = 0

    pool = Pool(processes=len(distributions))
    args = [conf, distributions, input_tensor, in_size, out_size, model, layer]
    for idx, cos_sim in enumerate(pool.imap_unordered(get_cosine_similarity, product(args))):
        if(cos_sim > max_match):
            max_match = cos_sim
            max_match_idx = idx

    return max_match_idx

def get_cosine_similarity(conf, distribution, input_tensor, in_size, out_size, model, layer):
    """
    Get cosine similarity score
    conf: Configuration for InGAN
    distribution: current distribution
    input_tensor: Input image tensor
    in_size: Input image size
    out_size: Output image size
    model: Model for feature extraction
    layer: Layer from which to take feature map
    """
    # Make current distribution as test distribution for InGAN
    conf.test_params_path = distribution
    gan = InGAN(conf)
    gan.resume(conf.test_params_path, test_flag=True)

    # Generate output image using generator
    output_tensor, _, _ = gan.test(input_tensor=input_tensor,
                                   input_size=in_size,
                                   output_size=out_size,
                                   rand_affine=None,
                                   run_d_pred=False,
                                   run_reconstruct=False)

    # Create both input image and image generated from InGAN
    output_image = util.tensor2im(output_tensor[1])
    input_image = util.tensor2im(input_tensor)

    # Extract features from both images
    feat1 = get_feature_vector(input_image, model, layer)
    feat2 = get_feature_vector(output_image, model, layer)

    # Using PyTorch Cosine Similarity
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    cos_sim = cos(feat1.unsqueeze(0),
                  feat2.unsqueeze(0))
    print('\nCosine similarity {0} : {1}\n'.format(distribution, cos_sim))
    return cos_sim

def initiate_sequential_network(distributions, input_tensor):
    """
    Get index of best matching distribution with
    the given input_tensor
    distributions: All distributions
    input_tensor: tensor of input image
    """
    # Calculate training_data and output image size
    in_size = input_tensor.shape[2:]
    out_size = (in_size[0], in_size[0])

    # Read configuration
    conf = Config().parse(create_dir_flag=False)

    # Load pretrained Resnet50 model for extracting features
    model = models.resnet50(pretrained=True)

    # Get the layer from where we will get features
    layer = model._modules.get('avgpool')
    model.eval()

    max_match = 0
    max_match_idx = 0
    for idx, distribution in enumerate(distributions):
        # Make current distribution as test distribution for InGAN
        conf.test_params_path = distribution
        gan = InGAN(conf)
        gan.resume(conf.test_params_path, test_flag=True)

        # Generate output image using generator
        output_tensor, _, _ = gan.test(input_tensor=input_tensor,
                                       input_size=in_size,
                                       output_size=out_size,
                                       rand_affine=None,
                                       run_d_pred=False,
                                       run_reconstruct=False)

        # Create both input image and image generated from InGAN
        output_image = util.tensor2im(output_tensor[1])
        input_image = util.tensor2im(input_tensor)

        # Extract features from both images
        feat1 = get_feature_vector(input_image, model,layer)
        feat2 = get_feature_vector(output_image, model, layer)

        # Using PyTorch Cosine Similarity
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        cos_sim = cos(feat1.unsqueeze(0),
                      feat2.unsqueeze(0))
        if cos_sim > max_match:
            max_match = cos_sim
            max_match_idx = idx
        print('\nCosine similarity {0} : {1}\n'.format(distribution, cos_sim))

    return max_match_idx

def start_testing():
    # load all pretrained distributions which are learned
    distributions = [join(DIST_PATH, f) for f in listdir(DIST_PATH) if isfile(join(DIST_PATH, f))]

    # Read the first image we find in folder for testing
    img_path = ''
    img_name = ''
    for file in listdir(TEST_PATH):
        img_path = join(TEST_PATH, file)
        img_name = file
        break

    # Input image tensor
    [input_tensor] = read_data(img_path, 8)

    best_match_idx = 0
    # Get index of best matching distribution from all learned distributions
    # Try running parallel network, if failed (on windows), then try sequential
    try:
        best_match_idx = initiate_parallel_network(distributions, input_tensor)
    except Exception:
        print("Switching to sequential")
        best_match_idx = initiate_sequential_network(distributions, input_tensor)

    # Load configuration again for generating images using InGAN
    # This time replace distribution with best found distribution
    conf = Config().parse(create_dir_flag=False)
    conf.name = 'TEST_' + img_name
    conf.output_dir_path = RESULT_PATH
    conf.output_dir_path = util.prepare_result_dir(conf)
    conf.test_params_path = distributions[best_match_idx]

    # Generate images using InGAN for input image
    gan = InGAN(conf)
    gan.resume(conf.test_params_path, test_flag=True)
    test.generate_collage_and_outputs(conf, gan, input_tensor)

    print('Completed image generation for the test image')


if __name__ == '__main__':
    start_testing()