import subprocess
import itertools
import operator

def most_common(list):
  # get an iterable of (item, iterable) pairs
  sorted_list = sorted((x, i) for i, x in enumerate(list))
  # print 'SL:', SL
  groups = itertools.groupby(sorted_list, key=operator.itemgetter(0))
  # auxiliary function to get "quality" for an item
  def _auxfun(g):
    item, iterable = g
    count = 0
    min_index = len(list)
    for _, where in iterable:
      count += 1
      min_index = min(min_index, where)
    # print 'item %r, count %r, minind %r' % (item, count, min_index)
    return count, -min_index
  # pick the highest-count/earliest item
  return max(groups, key=_auxfun)[0]

WAPITI_LOCATION = "wapiti-1.5.0/wapiti"
TEMPLATE_FILE = "generated_files/template"
DATASET = "atis.test.talil"
MODEL_FILE = "generated_files/modele"
LABEL_FILE = "eval_files/labels"

if __name__ == '__main__':
    for id in range(51):
        try:
            subprocess.check_call(
                WAPITI_LOCATION + ' label -m ' + MODEL_FILE + str(id) + ' <' + DATASET + ' > ' + LABEL_FILE + str(id), shell=True)
        except subprocess.CalledProcessError as e:
            print(e.returncode)
            print(e.cmd)
            print(e.output)

    cut = '2'
    i = 1
    while i < 51:
        cut+=','+str(i*2)
        i+=1

    try:
        subprocess.check_call(
            'paste eval_files/labels* | cut -f '+cut+'> votes',shell=True)
    except subprocess.CalledProcessError as e:
        print(e.returncode)
        print(e.cmd)
        print(e.output)

    with open('votes','r') as f:
        with open('deliberation','w+') as f2:
            for line in f:
                l = line.split()
                if len(l) == 0:
                    f2.write('\n')
                else:
                    f2.write(most_common(l)+'\n')


    try:
        subprocess.check_call(
            'paste atis.test.talil deliberation > vote_result',shell=True)
    except subprocess.CalledProcessError as e:
        print(e.returncode)
        print(e.cmd)
        print(e.output)

