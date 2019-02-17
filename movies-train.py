from mxnet import gluon, init, nd, autograd
from mxnet.gluon import data as gdata, loss as gloss, model_zoo
from mxnet.gluon import utils as gutils
import time
import mxnet as mx
import os

def train_aug_transform(data, label):
    data = data.astype('float32')/255
    augs = mx.image.CreateAugmenter(data_shape=(3,182,182),
                                    rand_crop=0.5, rand_mirror=True, inter_method=10,
                                    brightness=0.125, contrast=0.125, saturation=0.125,
                                    pca_noise=0.02, mean=mx.nd.array([0.485, 0.456, 0.406]), 
                                    std=mx.nd.array([0.229, 0.224, 0.225]))
    for aug in augs:
        data = aug(data)
    data = nd.transpose(data, (2,0,1))
    return data, label

def valid_aug_transform(data, label):
    data = data.astype('float32')/255
    augs = mx.image.CreateAugmenter(data_shape=(3,182,182),
                                    mean=mx.nd.array([0.485, 0.456, 0.406]), 
                                    std=mx.nd.array([0.229, 0.224, 0.225]))
    for aug in augs:
        data = aug(data)
    data = nd.transpose(data, (2,0,1))
    return data, label

def _get_batch(batch, ctx):
    """Return features and labels on ctx."""
    features, labels = batch
    if labels.dtype != features.dtype:
        labels = labels.astype(features.dtype)
    return (gutils.split_and_load(features, ctx),
            gutils.split_and_load(labels, ctx), features.shape[0])

def train_loop(train_iter, test_iter, net, loss, trainer, ctx, num_epochs):
    """Train and evaluate a model."""
    print('training on', ctx)
    if isinstance(ctx, mx.Context):
        ctx = [ctx]
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, m, start = 0.0, 0.0, 0, 0, time.time()
        for i, batch in enumerate(train_iter):
            Xs, ys, batch_size = _get_batch(batch, ctx)
            ls = []
            with autograd.record():
                y_hats = [net(X) for X in Xs]
                ls = [loss(y_hat, y) for y_hat, y in zip(y_hats, ys)]
            for l in ls:
                l.backward()
            trainer.step(batch_size)
            train_l_sum += sum([l.sum().asscalar() for l in ls])
            n += sum([l.size for l in ls])
            train_acc_sum += sum([(y_hat.argmax(axis=1) == y).sum().asscalar()
                                 for y_hat, y in zip(y_hats, ys)])
            m += sum([y.size for y in ys])
        test_acc, test_loss = evaluate_accuracy(test_iter, net, ctx)
        print('epoch %d, train loss %.4f, test loss %.4f, train acc %.3f, test acc %.3f, '
              'time %.1f sec'
              % (epoch + 1, train_l_sum / n, test_loss, train_acc_sum / m, test_acc,
                 time.time() - start))
    return net

def evaluate_accuracy(data_iter, net, ctx=[mx.cpu()]):
    """Evaluate accuracy of a model on the given data set."""
    if isinstance(ctx, mx.Context):
        ctx = [ctx]
    loss_sum, acc_sum, n, m = 0.0, 0.0, 0, 0
    loss = gloss.SoftmaxCrossEntropyLoss()
    for batch in data_iter:
        Xs, ys, batch_size = _get_batch(batch, ctx)
        y_hats = [net(X) for X in Xs]
        ys = [y.astype('float32') for y in ys]
        ls = [loss(y_hat, y) for y_hat, y in zip(y_hats, ys)]
        loss_sum += sum([l.sum().asscalar() for l in ls])
        n += sum([l.size for l in ls])
        acc_sum += sum([(y_hat.argmax(axis=1) == y).sum().asscalar() for y_hat, y in zip(y_hats, ys)])
        m += sum([y.size for y in ys])        
        
    return acc_sum / m, loss_sum / n

def get_model(what):
    if what == 'resnet34':
        pretrained_net = model_zoo.vision.resnet34_v2(pretrained=True)
        finetune_net = model_zoo.vision.resnet34_v2(classes=2)
    
    finetune_net.features = pretrained_net.features
    finetune_net.output.initialize(init.Xavier())
    return finetune_net

def train(channel_input_dirs, hyperparameters):
    
    # Retrieve the hyperparameters
    batch_size = hyperparameters.get('batch_size', 128)
    num_epochs = hyperparameters.get('epochs', 5)
    learning_rate = hyperparameters.get('learning_rate', 0.01)
    which_net = hyperparameters.get('which_net', 'resnet_34')
    freeze = hyperparameters.get('freeze', False)
    
    # Prepare the data
    data_dir = channel_input_dirs['training']
    training_dataset = mx.gluon.data.vision.ImageRecordDataset(data_dir + '/train_bi.rec', transform=train_aug_transform)
    validation_dataset = mx.gluon.data.vision.ImageRecordDataset(data_dir + '/valid_bi.rec', transform=valid_aug_transform)
    
    train_iter = mx.gluon.data.DataLoader(training_dataset, batch_size=batch_size, shuffle=True)
    test_iter = mx.gluon.data.DataLoader(validation_dataset, batch_size=batch_size)

    # Create the model
    net = get_model(which_net)

    ctx = mx.gpu()
    net.collect_params().reset_ctx(ctx)
    net.hybridize()
    loss = gloss.SoftmaxCrossEntropyLoss()
    
    if freeze:
        params = net.output.collect_params()
    else:
        net.output.collect_params().setattr('lr_mult', 100)
        params = net.collect_params()
        learning_rate /= 100
    
    trainer = gluon.Trainer(params, 'adam', {'learning_rate': learning_rate})
    
    return train_loop(train_iter, test_iter, net, loss, trainer, ctx, num_epochs)

def save(net, model_dir):
    net.save_params('%s/model.params' % model_dir)
    
def model_fn(model_dir):
    net = models.get_model('resnet34_v2', ctx=mx.cpu(), pretrained=False, classes=2)
    net.load_params('%s/model.params' % model_dir, ctx=mx.cpu())
    return net

def transform_fn(net, data, input_content_type, output_content_type):
    data = mx.nd.array(json.loads(data))
    data = data.astype('float32')/255
    augs = mx.image.CreateAugmenter(data_shape=(3,64,64),
                                    mean=mx.nd.array([0.485, 0.456, 0.406]), 
                                    std=mx.nd.array([0.229, 0.224, 0.225]))
    for aug in augs:
        data = aug(data)
    data = nd.transpose(data, (2,0,1))
    data = nd.expand_dims(data, axis=0)
    
    outputs = sig(net(data))
    nz = outputs.asnumpy().squeeze().nonzero()[0]
    classes = np.array(['Action', 'Adult', 'Adventure', 'Animation', 'Biography', 'Comedy',
       'Crime', 'Documentary', 'Drama', 'Family', 'Fantasy', 'Film-Noir',
       'Game-Show', 'History', 'Horror', 'Music', 'Musical', 'Mystery',
       'News', 'Reality-TV', 'Romance', 'Sci-Fi', 'Short', 'Sport',
       'Talk-Show', 'Thriller', 'War', 'Western'], dtype=object)
    
    response = json.dumps(classes[nz].tolist())
    return response, output_content_type
