import sys
import torch
from torch.autograd import Variable
import utils
import dataset
from PIL import Image

import models.crnn as crnn



model_path = sys.argv[1]
img_path = sys.argv[2]
alphabet = '0123456789/:'

model = crnn.CRNN(32, 1, len(alphabet) + 1, 256)
if torch.cuda.is_available():
    model = model.cuda()
print('loading pretrained model from %s' % model_path)
state_dict = torch.load(model_path)
state_dict = dict([(k[len('module.'):] if k.startswith('module.') else k, v) for k, v in state_dict.items()])
model.load_state_dict(state_dict)

converter = utils.strLabelConverter(alphabet)

transformer = dataset.resizeNormalize((100, 32))
image = Image.open(img_path).convert('L')
image = transformer(image)
if torch.cuda.is_available():
    image = image.cuda()
image = image.view(1, *image.size())
image = Variable(image)

model.eval()
preds = model(image)

_, preds = preds.max(2)
# no need to squeeze
# preds = preds.squeeze(2)
preds = preds.transpose(1, 0).contiguous().view(-1)

preds_size = Variable(torch.IntTensor([preds.size(0)]))
raw_pred = converter.decode(preds.data, preds_size.data, raw=True)
sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
print('%-20s => %-20s' % (raw_pred, sim_pred))