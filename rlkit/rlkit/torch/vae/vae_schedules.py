def always_train(epoch):
    return True, 300


def pretrain(epoch, batches):
    if epoch < 0:
        return True, batches
    else:
        return False, 0


def pretrain_1k(epoch):
    return pretrain(epoch, 1000)


def pretrain_10k(epoch):
    return pretrain(epoch, 10000)


def pretrain_20k(epoch):
    return pretrain(epoch, 20000)


def pretrain_30k(epoch):
    return pretrain(epoch, 30000)


def pretrain_40k(epoch):
    return pretrain(epoch, 40000)


def pretrain_50k(epoch):
    return pretrain(epoch, 50000)


def custom_schedule(epoch):
    if epoch < 10:
        return True, 1000
    elif epoch < 300:
        return True, 200
    else:
        return epoch % 3 == 0, 200


def custom_schedule_2(epoch):
    if epoch < 10:
        return True, 1000
    elif epoch < 100:
        return True, 200
    else:
        return epoch % 2 == 0, 200


def custom_schedule_3(epoch):
    if epoch < 10:
        return True, 2000
    elif epoch < 100:
        return True, 200
    else:
        return epoch % 2 == 0, 200


def custom_schedule_4(epoch):
    """For 120 epochs each with 150 rollouts each"""
    if epoch < 4:
        return True, 5000
    elif epoch < 24:
        return True, 1000
    else:
        return True, 500


def every_other(epoch):
    return epoch % 2 == 0, 400


def every_three(epoch):
    return epoch % 3 == 0, 600


def every_three_a_lot(epoch):
    return epoch % 3 == 0, 1200


def every_six(epoch):
    return epoch % 6 == 0, 1200


def every_six_less(epoch):
    return epoch % 6 == 0, 600


def every_six_much_less(epoch):
    return epoch % 6 == 0, 300


def every_ten(epoch):
    return epoch % 10 == 0 or epoch == 5, 1000


def every_twenty(epoch):
    return epoch % 10 == 0 or epoch == 5 or epoch == 10, 1000


def never_train(epoch):
    return False, 0
