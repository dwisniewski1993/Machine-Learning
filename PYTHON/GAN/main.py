from GenerativeAdversarialNetwork.GANModel import GAN


def main():
    gan_model = GAN()
    gan_model.load_data()
    gan_model.train_model(epochs=500, batch_size=128)
