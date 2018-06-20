import torch
from torch import nn

def dirtt():
    T = tb.utils.TensorDict(dict(
        sess = tf.Session(config=tb.growth_config()),
        src_x = placeholder((None, 32, 32, 3)),
        src_y = placeholder((None, args.Y)),
        trg_x = placeholder((None, 32, 32, 3)),
        trg_y = placeholder((None, args.Y)),
        test_x = placeholder((None, 32, 32, 3)),
        test_y = placeholder((None, args.Y)),
    ))
    # Supervised and conditional entropy minimization
    src_e = encoder(src_x)
    trg_e = encoder(trg_x)
    src_p = classifier(src_e)
    trg_p = classifier(trg_e)

    lossfunc_ce  = nn.CrossEntropy()
    lossfunc_nll = nn.Sequential(nn.LogSoftmax(), nn.NLLLoss())

    # src_y is the ground truth label distribution. use the normal ce loss
    loss_src_class = lossfunc_ce(src_p, src_y)
    # cross entropy loss
    loss_trg_cent  = lossfunc_nll(trg_p, trg_p)

    # Domain confusion
    if args.dw > 0 and args.dirt == 0:
        real_logit = discriminator(src_e)
        fake_logit = discriminator(trg_e)

        loss_disc = 0.5 * (
            lossfunc_ce(real_logit, torch.ones_like(real_logit)) +
            lossfunc_ce(fake_logit, torch.zeros_like(fake_logit))
        loss_domain = 0.5 * (
            lossfunc_ce(real_logit, torch.zeros_like(real_logit)) +
            lossfunc_ce(fake_logit, torch.ones_like(fake_logit))

    else:
        loss_disc = 0
        loss_domain = 0

    # Virtual adversarial training (turn off src in non-VADA phase)
    loss_src_vat = vat_loss(src_x, src_p, model) if args.sw > 0 and args.dirt == 0 else 0
    loss_trg_vat = vat_loss(trg_x, trg_p, model) if args.tw > 0 else 0


def train():

    # Evaluation (EMA)
    ema = optim.ExponentialMovingAverage(decay=0.998)
    ema_p = model(test_x, train=False)

    # Teacher model (a back-up of EMA model)
    teacher_p = model(test_x, train=False)

    # Accuracies
    src_acc = acc_func(src_p, src_y)
    trg_acc = acc_func(trg_p, trg_y)
    ema_acc = acc_func(ema_p, test_y)

    # Optimizer
    dw = constant(args.dw) if args.dirt == 0 else constant(0)
    cw = constant(1)       if args.dirt == 0 else constant(args.bw)
    sw = constant(args.sw) if args.dirt == 0 else constant(0)
    tw = constant(args.tw)
    loss_main = (dw * loss_domain +
                 cw * loss_src_class +
                 sw * loss_src_vat +
                 tw * loss_trg_cent +
                 tw * loss_trg_vat)
    var_main = tf.get_collection('trainable_variables', 'class')
    train_main = tf.train.AdamOptimizer(args.lr, 0.5).minimize(loss_main, var_list=var_main)
    train_main = tf.group(train_main, ema_op)

    if args.dw > 0 and args.dirt == 0:
        var_disc = tf.get_collection('trainable_variables', 'disc')
        train_disc = tf.train.AdamOptimizer(args.lr, 0.5).minimize(loss_disc, var_list=var_disc)
    else:
        train_disc = constant(0)

    T.ops_print = [c('disc'), loss_disc,
                   c('domain'), loss_domain,
                   c('class'), loss_src_class,
                   c('cent'), loss_trg_cent,
                   c('trg_vat'), loss_trg_vat,
                   c('src_vat'), loss_src_vat,
                   c('src'), src_acc,
                   c('trg'), trg_acc]
    T.ops_disc = [summary_disc, train_disc]
    T.ops_main = [summary_main, train_main]
    T.fn_ema_acc = fn_ema_acc
    T.teacher = teacher
    T.update_teacher = update_teacher

    return T
