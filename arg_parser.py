import argparse


def arguments():
    # Instantiate the parser
    parser = argparse.ArgumentParser(description='Optional app description')

    # Optional argument
    parser.add_argument('--epochs', type=int,
                        help='amount of epochs for training')

    parser.add_argument('--folds', type=int,
                        help='amount of folds for k-fold training')


    parser.add_argument('--samples', type=int,
                        help='amount of samples to take (default is all)')

    parser.add_argument('--batch_size', type=int,
                    help='batch size')

    # Switch
    parser.add_argument('--no_class_weights', action='store_false',
                        help='A boolean switch')

    return parser.parse_args()

