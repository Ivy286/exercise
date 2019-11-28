import argparse
parser = argparse.ArgumentParser(description="calculate X to the power of Y")
parser.add_argument("x", type=int, help="the base")
parser.add_argument("y", type=int, help="the exponent")
parser.add_argument('-v', "--verbosity", help="increase output verbosity", action='count', default=0)
args = parser.parse_args()
answer = args.x**args.y
if args.verbosity >= 2:
    print("Running '{}'".format(__file__))
elif args.verbosity >= 1:
    print("{}^{} ==".format(args.x, args.y))
print(answer)
