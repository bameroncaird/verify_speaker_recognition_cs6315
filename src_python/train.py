from data import get_datasets
from models import simple_mlp_model, conv_1d_model, conv_2d_model
import math


if __name__ == '__main__':
    
    batch_size = 64
    train_ds, dev_ds, test_ds = get_datasets('libri100', batch_size)
    n_train = 1000
    n_dev = 150
    n_test = 200

    # iterate over training dataset
    # the dataset is infinite now but you can still check shapes
    # for epoch in range(1):
    #     for data_batch, label_batch in train_ds:
    #         print(data_batch.shape, label_batch.shape)

    model = simple_mlp_model(adversarial=True)
    #model = conv_1d_model()
    #model = conv_2d_model()
    print(model)
    #model.build((None, 2000))
    #model.summary()
    history = model.fit(
        x=train_ds,
        batch_size=batch_size,
        epochs=30,
        steps_per_epoch=math.ceil(n_train / batch_size),
        callbacks=None,
        validation_data=dev_ds,
        validation_steps=math.ceil(n_dev / batch_size),
        workers=1,
        use_multiprocessing=False,
    )
    print(history.history.keys())

    model.evaluate(test_ds, steps=math.ceil(n_test / batch_size))

    model.save('../saved_models/mlp_adversarial_config3_pgd_10iters')
