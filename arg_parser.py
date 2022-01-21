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

    parser.add_argument('--timepoints', type=int,
                help='timepoints')

    parser.add_argument('--name', type=str,
                    help='filename_suffix')

    parser.add_argument('--model', type=str,
                help='bl/lstm/trans')

    # Switch
    parser.add_argument('--no_class_weights', action='store_false',
                        help='Dont use weighted loss')

    parser.add_argument('--no_process_data', action='store_false',
                        help='Dont pre-process data')

    return parser.parse_args()
