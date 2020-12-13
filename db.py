import datetime

from mongoengine import connect, Document
from mongoengine import BooleanField, IntField, FloatField, StringField, ListField, DateTimeField
connect('results')


class Post(Document):
    # run number
    run = IntField(unique=True, required=True)
    
    # descriptive model name
    model_name = StringField(required=True)
    
    # config parameters (only relevant fields to a particular run are filled)
    nn_type = StringField(required=True)
    params = ListField()
    
    activation = StringField(required=True)
    norm = StringField(required=True)
    weights_init = StringField(required=True)
    
    dataset = StringField(required=True)
    normalize_input = BooleanField(required=True)
    
    epochs = IntField(required=True)
    batch_size = IntField(required=True)
    optim = StringField(required=True)
    lr = FloatField(required=True)
    shuffle = BooleanField(required=True)
    momentum = FloatField()
    nesterov = BooleanField()
    beta1 = FloatField()
    beta2 = FloatField()
    
    num_workers = IntField()
    device = StringField(choices=('cuda', 'cpu'))
    random_seed = IntField()
    
    # metrics
    train_loss_avg = ListField()
    train_loss_std = ListField()
    val_loss_avg = ListField()
    val_loss_std = ListField()
    
    train_acc = ListField()
    val_acc = ListField()
    
    best_epoch_train_loss = IntField()
    best_epoch_train_acc = IntField()
    best_epoch_val_loss = IntField()
    best_epoch_val_acc = IntField()
    
    train_loss_at_best_train_loss = FloatField()
    train_acc_at_best_train_loss = FloatField()
    val_loss_at_best_train_loss = FloatField()
    val_acc_at_best_train_loss = FloatField()
    
    train_loss_at_best_train_acc = FloatField()
    train_acc_at_best_train_acc = FloatField()
    val_loss_at_best_train_acc = FloatField()
    val_acc_at_best_train_acc = FloatField()
    
    train_loss_at_best_val_loss = FloatField()
    train_acc_at_best_val_loss = FloatField()
    val_loss_at_best_val_loss = FloatField()
    val_acc_at_best_val_loss = FloatField()
    
    train_loss_at_best_val_acc = FloatField()
    train_acc_at_best_val_acc = FloatField()
    val_loss_at_best_val_acc = FloatField()
    val_acc_at_best_val_acc = FloatField()
    
    train_entropy_avg = ListField()
    train_entropy_std = ListField()
    val_entropy_avg = ListField()
    val_entropy_std = ListField()
    
    entropy_rand_avg = ListField()
    entropy_rand_std = ListField()
    
    # timestamp
    timestamp = DateTimeField(default=datetime.datetime.utcnow)


def get_last(field):
    if Post.objects.first():
        post = Post.objects.order_by('-timestamp').limit(1).first()
        return post[field]
    else:
        return 0
