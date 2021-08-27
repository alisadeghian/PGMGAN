from gan_training.logger import Logger
from os import path 

# Logger(log_dir=path.join(out_dir, 'logs'),
#         img_dir=path.join(out_dir, 'imgs'),
#         monitoring=config['training']['monitoring'],
#         monitoring_dir=path.join(out_dir, 'monitoring'))

out_dir = "output/cifar/scan_guide42_biggan"

logger = Logger(log_dir=path.join(out_dir, 'logs'),
                    img_dir=path.join(out_dir, 'imgs'),
                    monitoring=None,
                    monitoring_dir=None)

logger.load_stats('stats_00250000.p')

for i in range(600):
    print(logger.stats['losses']['guide_resnet_loss'][-i])


