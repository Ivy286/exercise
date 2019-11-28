import argparse

parser = argparse.ArgumentParser()
parser.description='give me two number, I can give you a multipy'
parser.add_argument("-a", "--ParA", help='I am A', type=int)
parser.add_argument("-b", "--ParB", help='I am B', type=int)

args = parser.parse_args()
if args.ParA:
    print('receive A, is', args.ParA)
if args.ParB:
    print('receive B, is', args.ParB)
if args.ParA and args.ParB:
    print('haha, mulitipy is', args.ParA*args.ParB)
