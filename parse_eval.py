#!/usr/bin/python3
import sys
import subprocess
#https://stackoverflow.com/questions/13332268/how-to-use-subprocess-command-with-pipes

#Currently the model file must already exist
#TODO generate model file from individual



WAPITI_LOCATION = "wapiti-1.5.0/wapiti"
TEMPLATE_FILE = "template"
DATASET = "atis.train"
MODEL_FILE = "modele"
LABEL_FILE = "labels"
EVAL_FILE = "eval"

try:
    subprocess.check_call(WAPITI_LOCATION+' train -p '+TEMPLATE_FILE+' -t 8 '+DATASET+' '+MODEL_FILE, shell = True)
except subprocess.CalledProcessError as e:
    print(e.returncode)
    print(e.cmd)
    print(e.output)

try:
    subprocess.check_call(WAPITI_LOCATION+' label -m '+MODEL_FILE+' <'+DATASET+' > '+LABEL_FILE, shell = True)
except subprocess.CalledProcessError as e:
    print(e.returncode)
    print(e.cmd)
    print(e.output)

try:
    subprocess.check_call('cat '+LABEL_FILE+' | perl evaluation.pl > '+EVAL_FILE, shell = True)
except subprocess.CalledProcessError as e:
    print(e.returncode)
    print(e.cmd)
    print(e.output)

with open(EVAL_FILE, "r") as f:
    for i, line in enumerate(f):
        if(i==1):
            l = line.split()
            accuracy=l[1]
            precision=l[3]
            recall=l[5]
            f1=l[7]
            break
print("Score : Accuracy="+accuracy+"; Precision="+precision+"; Recall="+recall+"; F-measure="+f1)
