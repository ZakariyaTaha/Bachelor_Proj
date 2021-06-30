import argparse
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
import time
from dataloader import KITTILoader as DA
import utils.logger as logger
import torch.backends.cudnn as cudnn
import pdb
import models.anynet
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image as Image
import cv2

parser = argparse.ArgumentParser(description='Anynet fintune on KITTI')
parser.add_argument('--maxdisp', type=int, default=192,
                    help='maxium disparity')
parser.add_argument('--loss_weights', type=float, nargs='+', default=[0.25, 0.5, 1., 1.])
parser.add_argument('--max_disparity', type=int, default=192)
parser.add_argument('--maxdisplist', type=int, nargs='+', default=[12, 3, 3])
parser.add_argument('--datatype', default='2015',
                    help='datapath')
parser.add_argument('--datapath', default=None, help='datapath')
parser.add_argument('--epochs', type=int, default=300,
                    help='number of epochs to train')
parser.add_argument('--train_bsize', type=int, default=6,
                    help='batch size for training (default: 6)')
parser.add_argument('--test_bsize', type=int, default=8,
                    help='batch size for testing (default: 8)')
parser.add_argument('--save_path', type=str, default='results/finetune_anynet',
                    help='the path of saving checkpoints and log')
parser.add_argument('--resume', type=str, default=None,
                    help='resume path')
parser.add_argument('--lr', type=float, default=5e-4,
                    help='learning rate')
parser.add_argument('--with_spn', action='store_true', help='with spn network or not')
parser.add_argument('--print_freq', type=int, default=5, help='print frequence')
parser.add_argument('--init_channels', type=int, default=1, help='initial channels for 2d feature extractor')
parser.add_argument('--nblocks', type=int, default=2, help='number of layers in each stage')
parser.add_argument('--channels_3d', type=int, default=4, help='number of initial channels 3d feature extractor ')
parser.add_argument('--layers_3d', type=int, default=4, help='number of initial layers in 3d network')
parser.add_argument('--growth_rate', type=int, nargs='+', default=[4,1,1], help='growth rate in the 3d network')
parser.add_argument('--spn_init_channels', type=int, default=8, help='initial channels for spnet')
parser.add_argument('--start_epoch_for_spn', type=int, default=121)
parser.add_argument('--pretrained', type=str, default='results/pretrained_anynet/checkpoint.tar',
                    help='pretrained model path')
parser.add_argument('--split_file', type=str, default=None)
parser.add_argument('--evaluate', action='store_true')


args = parser.parse_args()



def main():

    from dataloader import diy_dataset as ls

    if not os.path.isdir(args.datapath + '/log_saves'):
        os.makedirs(args.datapath + '/log_saves')
        
    log = logger.setup_logger(args.datapath + '/log_saves' + '/training.log')

    train_left_img, train_right_img, train_left_disp, test_left_img, test_right_img, test_left_disp = ls.dataloader(
       args.datapath,log, None)

    TestImgLoader = torch.utils.data.DataLoader(
        DA.myImageFloder(test_left_img, test_right_img, test_left_disp, False),
        batch_size=1, shuffle=False, num_workers=4, drop_last=False)

    model = models.anynet.AnyNet(args) ##.cuda()
    model = nn.DataParallel(model).cuda()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))

    checkpoint = torch.load(args.pretrained)
    #print(checkpoint.keys())
    model.load_state_dict(checkpoint['state_dict'], strict = False)
    ## optimizer.load_state_dict(checkpoint['optimizer'])
    epoch = checkpoint['epoch']

    model.eval()   
    
    stages = 3 + args.with_spn
    length_loader = len(TestImgLoader)

    for batch_idx, (imgL, imgR, disp_L) in enumerate(TestImgLoader):
        imgL = imgL.float().cuda()
        imgR = imgR.float().cuda()
        disp_L = disp_L.float().cuda()

        with torch.no_grad():
            outputs = model(imgL, imgR)

        log.info(np.shape(np.array(outputs)))

        output_0 = outputs[0].squeeze()
        output_1 = outputs[1].squeeze()
        output_2 = outputs[2].squeeze()
        log.info("output_0 : {}".format(output_0))

        
        Image.fromarray(output_0.cpu().numpy()).convert('L').save('withkitti_new_data_0.png')
        Image.fromarray(output_1.cpu().numpy()).convert('L').save('withkitti_new_data_1.png')
        Image.fromarray(output_2.cpu().numpy()).convert('L').save('withkitti_new_data_2.png')

        '''
        plt.imshow(output_0.cpu().numpy())
        plt.savefig('fig_0.png')
        plt.imshow(output_1.cpu().numpy())
        plt.savefig('fig_1.png')
        plt.imshow(output_2.cpu().numpy())
        plt.savefig('fig_2.png')
        return
        '''
if __name__ == '__main__':
    main()


