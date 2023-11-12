#! /opt/anaconda3/bin/python

import os
import sys

print (sys.argv)
if len(sys.argv) > 1:
    commit_ = str(sys.argv[1])
else:
    commit_ = "'add: new commitment'"

os.system('git add .')
os.system('git commit -m' + commit_)

os.system('git push -f')