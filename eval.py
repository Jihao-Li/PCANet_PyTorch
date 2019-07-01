import os
import sys
import time
import glob
import torch
import utils
import logging
import argparse
from dataset_mnist import load_test_mnist
from sklearn.metrics import accuracy_score


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser("PCANet")
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--dataset_name', type=str, default='mnist', help='mnist or cifar10')
parser.add_argument('--dataset_path', type=str, default='/dataset/', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--stages', type=int, default=2, help='the number of stages')
parser.add_argument('--pretrained_path', type=str, default='search-EXP-20190701-193327', help='pretrained_path')
parser.add_argument('--log_freq', type=int, default=30, help='record log frequency')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
args = parser.parse_args()

args.save = 'eval-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

# generate log file
log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)


def main():
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    logging.info('gpu device = %d' % args.gpu)
    logging.info("args = %s", args)

    pcanet = utils.load_model(args.pretrained_path + "/pcanet.pkl")
    classifier = utils.load_model(args.pretrained_path + "/classifier.pkl")
    logging.info("load PCANet and SVM completely")

    test_queue, num_test = load_test_mnist(args)       # load dataset
    logging.info("load testing dataset completely")

    with torch.no_grad():
        num_of_correct_samples = 0
        for global_step, (test_images, test_labels) in enumerate(test_queue):
            batch_size = test_images.shape[0]
            batch_features = test_images.cuda()

            # execute convolution in different stages
            for stage in range(args.stages):
                batch_features = pcanet.pca_conv(batch_features, pcanet.kernel[stage])

            # build binary quantization mapping and generate histogram
            decimal_features = pcanet.binary_mapping(batch_features, stage)
            final_features = pcanet.generate_histogram(decimal_features)

            # calculate the rate of correct classification
            final_features = final_features.cpu()
            predict_class = classifier.predict(final_features)
            batch_accuracy = accuracy_score(predict_class, test_labels)

            if global_step % args.log_freq == 0:
                logging.info("global_step %d, stage %d, batch accuracy %f" % (global_step, stage, batch_accuracy))

            batch_num_of_correct_samples = utils.total_accuracy(predict_class, test_labels)
            num_of_correct_samples += batch_num_of_correct_samples

        logging.info("total accuracy %f" % (num_of_correct_samples / num_test))
        logging.info("test completely")


if __name__ == "__main__":
    main()
