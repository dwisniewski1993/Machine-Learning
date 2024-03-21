from GenerativeAdversarialNetwork.GANModel import GAN
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def main():
    gan_model = GAN()
    gan_model.load_data()
    gan_model.train_model(epochs=500, batch_size=128)


if __name__ == '__main__':
    main()
