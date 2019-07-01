import math
import torch
import numpy as np
import torch.nn.functional as F
from sklearn.decomposition import IncrementalPCA


class PCANet(object):
    """ define PCANet class"""
    def __init__(self, stages, filter_shape, stages_channels, block_size, block_overlap):
        self.stages = stages
        self.filter_shape = filter_shape
        self.stages_channels = stages_channels

        self.block_size = block_size
        self.block_overlap = block_overlap       # the rate of overlap between blocks
        self.grey_level = pow(2, self.stages_channels[-1])
        self.kernel = {}        # convolution kernels in different stages

        self.pca = {}
        for i, output_channel in enumerate(stages_channels):
            self.pca[i] = IncrementalPCA(n_components=output_channel)

    def unrolled_stage(self, batch_images, stage):
        batch_size = batch_images.shape[0]
        # TODO　for channel in channels:
        num_paddings = (self.filter_shape[stage] - 1) // 2     # (rawH - filter + 2*padding)/stride + 1 = processedH
        patches = F.unfold(batch_images, kernel_size=self.filter_shape[stage], padding=num_paddings)
        patches = patches.reshape(batch_size, self.filter_shape[stage] * self.filter_shape[stage], -1)
        patches = self.patch_mean_removal(patches)
        patch_matrix = patches.reshape(-1, self.filter_shape[stage] * self.filter_shape[stage])
        patch_matrix = patch_matrix.cpu()
        self.pca[stage].partial_fit(patch_matrix)

    def patch_mean_removal(self, patches):
        patches_mean = torch.mean(patches, dim=1, keepdim=True)
        patches = patches - patches_mean
        return patches

    def eigenvector_to_kernel(self, stage):
        eigenvector = self.pca[stage].components_
        eigenvector = eigenvector.astype(np.float32)
        kernel = eigenvector.reshape(self.stages_channels[stage], -1,
                                     self.filter_shape[stage], self.filter_shape[stage])
        self.kernel[stage] = torch.from_numpy(kernel).cuda()
        return self.kernel[stage]

    def pca_conv(self, images, kernel):
        """
        'SAME' mode convolution
        :param images: 
        :param kernel: 
        :return: 
        """
        batch_size = images.shape[0]
        kernel_hw = kernel.shape[2]
        iamges_h = images.shape[2]
        images_w = images.shape[3]
        images = images.reshape(-1, 1, iamges_h, images_w)
        padding = (kernel_hw - 1) // 2
        features = F.conv2d(images, kernel, padding=padding)

        features_h = features.shape[2]
        features_w = features.shape[3]
        features = features.reshape(batch_size, -1, features_h, features_w)
        return features

    def binary_mapping(self, batch_features, stage):
        batch_size = batch_features.shape[0]
        features_h = batch_features.shape[2]
        features_w = batch_features.shape[3]

        binary_features = binarize(batch_features)       # step function
        binary_features = binary_features.reshape(-1, self.stages_channels[stage], features_h, features_w)

        decimal_features = binary_to_decimal(binary_features, self.stages_channels[stage])
        decimal_features = decimal_features.reshape(batch_size, -1, features_h, features_w)
        return decimal_features

    def generate_histogram(self, decimal_features):
        """
        
        :param decimal_features: 
        :return: 
        """
        batch_size = decimal_features.shape[0]
        stride = math.ceil((1 - self.block_overlap) * self.block_size)
        block_features = F.unfold(decimal_features, kernel_size=self.block_size, stride=stride)
        block_features = block_features.reshape(batch_size, -1, self.block_size * self.block_size)
        num_of_features = block_features.shape[1]

        final_features = torch.tensor([]).cuda()    # empty tensor
        for batch in range(batch_size):
            single_image_feature = torch.tensor([]).cuda()     # empty tensor

            for count in range(num_of_features):
                temp_feature = block_features[batch, count, :]
                temp_feature = temp_feature.cpu()
                feature_histogram = torch.histc(temp_feature, min=0.5, max=self.grey_level - 0.5, bins=self.grey_level)
                feature_histogram = feature_histogram.cuda()
                single_image_feature = torch.cat((single_image_feature, feature_histogram), dim=0)

            # final feature of a single image
            single_image_feature = single_image_feature.unsqueeze(0)
            # final feature of a batch of images
            final_features  = torch.cat((final_features, single_image_feature), dim=0)

        return final_features


def binarize(batch_features):
    """
    step function
    :param batch_features: 
    :return: 
    """
    return (torch.sign(batch_features) + 1.0) / 2.0


def binary_weight(channels):
    """
    generate binary power series
    :param channels: 
    :return: 
    """
    power_series = torch.arange(channels).float().cuda()
    power_series = torch.pow(2, power_series)
    return power_series


def binary_to_decimal(binary_features, channels):
    num_of_features = binary_features.shape[0]

    power_series = binary_weight(channels)       # generate binary weight
    power_series_shape = power_series.shape[0]
    power_series = power_series.reshape(1, power_series_shape, 1, 1)
    power_series = power_series.expand(num_of_features, power_series_shape, 1, 1)

    decimal_features = torch.sum(binary_features * power_series, dim=1)      # binary to decimal
    return decimal_features


if __name__ == "__main__":
    batch_features = torch.randn(2, 4, 3, 3)
    print(batch_features)
    batch_feature = binarize(batch_features)
    print(batch_features)

    channels = 4
    binary_features = binary_to_decimal(batch_features, channels)
    print(binary_features)
