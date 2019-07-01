import os
import sys
import time
import glob
import torch
import utils
import logging
import argparse
from pcanet import PCANet
from dataset_mnist import load_train_mnist
from sklearn.svm import LinearSVC, SVC
from tensorboardX import SummaryWriter
from torchvision.utils import make_grid


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser("PCANet")
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--dataset_name', type=str, default='mnist', help='mnist or cifar10')
parser.add_argument('--dataset_path', type=str, default='/dataset/', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--train_portion', type=float, default=1.0, help='portion of training data')
parser.add_argument('--stages', type=int, default=2, help='the number of stages')
parser.add_argument('--filter_shape', type=list, default=[7, 7], help='patch size')
parser.add_argument('--stages_channels', type=list, default=[8, 8], help='channels in different stages')
parser.add_argument('--block_size', type=int, default=7, help='the size of blocks')
parser.add_argument('--block_overlap', type=float, default=0.5, help='the rate of overlap between blocks')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--log_freq', type=int, default=40, help='record log frequency')
args = parser.parse_args()

args.save = 'search-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

# generate log file
log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)


CLASSES = 10      # 10 classes in both mnist and cifar


def main():
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    logging.info('gpu device = %d' % args.gpu)
    logging.info("args = %s", args)

    pcanet = PCANet(args.stages, args.filter_shape, args.stages_channels, args.block_size, args.block_overlap)
    train_queue, valid_queue = load_train_mnist(args)        # load dataset
    logging.info("load training dataset completely")
    total_train_labels = torch.tensor([]).long()

    writer = SummaryWriter(args.save)       # tensorboardX

    # extract feature from images
    with torch.no_grad():
        # first of all, generate eigenvector, and then execute convolution
        stage_save_path = args.save
        save_filename = utils.create_pickle_file_name(stage_save_path, 0)
        for global_step, (train_images, train_labels) in enumerate(train_queue):
            train_images = train_images.cuda()
            total_train_labels = torch.cat((total_train_labels, train_labels))
            utils.save_feature([train_images, train_labels], save_filename)
            pcanet.unrolled_stage(train_images, 0)

            if global_step % args.log_freq== 0:
                logging.info("init training global_step: %d" % global_step)
                # convert a batch of tensor into CHW format
                grid_images = make_grid(train_images, nrow=16, padding=5, pad_value=125)
                writer.add_image("raw_images_in_step_%d" % global_step, grid_images)

        total_features = torch.tensor([])       # empty tensor
        for stage in range(args.stages):
            logging.info('PCANet stage: %d' % stage)

            # transform eigenvector to convolution kernel
            kernel = pcanet.eigenvector_to_kernel(stage)

            load_filename = utils.create_pickle_file_name(stage_save_path, stage)
            if stage + 1 < args.stages:
                save_filename = utils.create_pickle_file_name(stage_save_path, stage + 1)

            load_filename_pointer = 0         # clear file object pointer
            for step in range(global_step + 1):
                train_images, train_labels, load_filename_pointer = \
                    utils.load_feature(load_filename, load_filename_pointer)
                batch_features = pcanet.pca_conv(train_images, kernel)
                if step % args.log_freq == 0:
                    # view the i-th image's feature map in a single batch
                    single_image_feature = utils.exchange_channel(batch_features[5])
                    grid_images = make_grid(single_image_feature, nrow=8, padding=5, pad_value=125)
                    writer.add_image("feature_image_in_step_%d_in_stage_%d" % (step, stage), grid_images)

                if stage + 1 < args.stages:
                    utils.save_feature([batch_features, train_labels], save_filename)
                    pcanet.unrolled_stage(batch_features, stage + 1)
                else:
                    decimal_features = pcanet.binary_mapping(batch_features, stage)
                    final_features = pcanet.generate_histogram(decimal_features)
                    final_features = final_features.cpu()
                    total_features = torch.cat((total_features, final_features), dim=0)

                if step % args.log_freq == 0:
                    logging.info("circulate training step: %d" % step)

            grid_kernels = make_grid(pcanet.kernel[stage], nrow=args.stages_channels[stage], padding=5, pad_value=125)
            writer.add_image("kernel_in_stage_%d" % stage, grid_kernels)

        writer.close()
        logging.info('extract feature completely, start training classifier')

        # train classifier
        classifier = LinearSVC()
        # classifier = SVC()
        # total_features = total_features.cpu()
        classifier.fit(total_features, total_train_labels)
        logging.info('classifier trained completely')

        # save model
        utils.save_model(pcanet, stage_save_path + "/pcanet.pkl")
        utils.save_model(classifier, stage_save_path + "/classifier.pkl")

        train_score = classifier.score(total_features, total_train_labels)
        logging.info("score of training is %s" % train_score)


if __name__ == "__main__":
    main()
