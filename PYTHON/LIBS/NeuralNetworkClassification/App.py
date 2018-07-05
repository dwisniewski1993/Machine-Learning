from NNC.NNC import NNC


def main():
    print("Neural Network for Classification")

    train_file = 'train/train.tsv'
    dev0_file = 'dev-0/in.tsv'
    testA_file = 'test-A/in.tsv'

    nnc = NNC(trainlocation=train_file, testalocation=testA_file, dev0location=dev0_file)
    nnc.train_model()
    nnc.dev0_output()
    nnc.testA_output()


if __name__ == "__main__":
    main()
