from train import Train

if __name__ == '__main__':
    train_model =  Train(lr_rate=0.0002, epoch=10, batch_size=5, input_size=(97,97))
    train_model.train(summary=False)