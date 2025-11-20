import re
from sage.all import matrix,QQ,sage_eval,gap

# load decomposition from https://fmm.univ-lille.fr/
def from_univ_lille(file,locals={}):
    t = open(file).read()
    t = re.search(r'TriadSet\((.*)\):',t).group(1)
    t = re.sub(r'Matrix\(\d, \d, (.*?)\)',r'\1',t)
    t = re.sub(r'Triad\((.*?)\)',r'\1',t)
    return [[matrix(QQ,m) for m in ms] for ms in sage_eval(t,locals=locals)]

def rec_to_dict(rec):
    return {str(field) : gap.get_record_element(rec,field) for field in rec.RecNames()}

def dict_to_rec(d):
    return gap('rec({})'.format(",".join([f'{k} := {v.name()}' for k,v in d.items()])))
