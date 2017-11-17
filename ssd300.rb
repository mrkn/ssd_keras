require 'pycall'

SSD_KERAS_DIR = File.dirname(__FILE__)
PyCall.import_module('sys').path.append(SSD_KERAS_DIR)
PICS_DIR = File.join(SSD_KERAS_DIR, 'pics')

require 'numpy'
np = Numpy
imagenet_utils = PyCall.import_module('keras.applications.imagenet_utils')
image = PyCall.import_module('keras.preprocessing.image')

require 'matplotlib/pyplot'
plt = Matplotlib::Pyplot

ssd = PyCall.import_module('ssd')
BBoxUtility = PyCall.import_module('ssd_utils').BBoxUtility

VOC_CLASSES = [
  'Aeroplane',
  'Bicycle',
  'Bird',
  'Boat',
  'Bottle',
  'Bus',
  'Car',
  'Cat',
  'Chair',
  'Cow',
  'Diningtable',
  'Dog',
  'Horse',
  'Motorbike',
  'Person',
  'Pottedplant',
  'Sheep',
  'Sofa',
  'Train',
  'Tvmonitor'
]
NUM_CLASSES = VOC_CLASSES.length + 1

INPUT_SHAPE = [300, 300, 3]
model = ssd.SSD300(INPUT_SHAPE, num_classes: NUM_CLASSES)
model.load_weights('weights_SSD300.hdf5', by_name: true)
bbox_util = BBoxUtility.new(NUM_CLASSES)

image_path = ARGV[0] || 'keras_input.jpg'
img = image.load_img(image_path, target_size: [300, 300])
img = image.img_to_array(img)
inputs = imagenet_utils.preprocess_input(np.array([img]))
preds = model.predict(inputs, batch_size: 1, verbose: 1)
results = bbox_util.detection_out(preds).to_a

# Parse output

all = PyCall::Slice.new(nil)
det_label = results[0][all, 0]
det_conf  = results[0][all, 1]
det_xmin  = results[0][all, 2]
det_ymin  = results[0][all, 3]
det_xmax  = results[0][all, 4]
det_ymax  = results[0][all, 5]

# Get detections with confidence higher than 0.6.
top_indices = det_conf.tolist.each_with_index.inject([]) do |ary, (conf, i)|
  ary.tap { ary << i if conf >= 0.6 }
end
top_indices = PyCall::List.new(top_indices)

top_conf = det_conf[top_indices]
top_label_indices = det_label[top_indices].tolist
top_xmin = det_xmin[top_indices]
top_ymin = det_ymin[top_indices]
top_xmax = det_xmax[top_indices]
top_ymax = det_ymax[top_indices]

colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist

plt.imshow(img / 255.0)
current_axis = plt.gca

top_conf.shape[0].times do |i|
  xmin = (top_xmin[i] * img.shape[1]).round.to_i
  ymin = (top_ymin[i] * img.shape[0]).round.to_i
  xmax = (top_xmax[i] * img.shape[1]).round.to_i
  ymax = (top_ymax[i] * img.shape[0]).round.to_i
  score = top_conf[i]
  label = top_label_indices[i].to_i
  label_name = VOC_CLASSES[label - 1]
  display_txt = '%0.2f, %s' % [score, label_name]
  coords = [[xmin, ymin], xmax-xmin+1, ymax-ymin+1]
  color = colors[label]
  current_axis.add_patch(plt.Rectangle.new(*coords, fill: false, edgecolor: color, linewidth: 2))
  current_axis.text(xmin, ymin, display_txt, bbox: {facecolor: color, alpha: 0.5})
end

plt.savefig('keras_output.png', dpi: 100)
