from argparse import ArgumentParser

parser = ArgumentParser()

parser.add_argument("--int_arg", default=10, help='int argument', type=int)
parser.add_argument("--float_arg", default=0.9, help='float argument', type=float)
parser.add_argument("--str_arg", default='this is a string', help='string argument')
parser.add_argument("--bool_store_true", action='store_true', help='enable True flag')
parser.add_argument("--bool_store_false", action='store_false', help='enable False flag')

args = parser.parse_args()
print('int argument =', args.int_arg)
print('float argument =', args.float_arg)
print('string argument =', args.str_arg)
print('bool(store_true) argument =', args.bool_store_true)
print('bool(store_false) argument =', args.bool_store_false)
