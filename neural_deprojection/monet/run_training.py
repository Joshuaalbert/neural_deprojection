import argparse,  sys
#imports


def main(arg0, arg1, arg2):
    # construct the model
    # run training
    # save model
    print(arg0, arg1, arg2)


def add_args(parser):
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument('--arg0', help='Obs number L*',default=0, type=int, required=False)
    parser.add_argument('--arg1', help='Obs number L*',default=0, type=int, required=False)
    parser.add_argument('--arg2', help='Obs number L*',default=0, type=int, required=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Runs simple training loop.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_args(parser)
    flags, unparsed = parser.parse_known_args()
    print("Running with:")
    for option, value in vars(flags).items():
        print("    {} -> {}".format(option, value))
    main(**vars(flags))