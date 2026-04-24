import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/kazaki/enae450-s26/final competition/enae450-final-competition/final-comp/install/final-comp'
