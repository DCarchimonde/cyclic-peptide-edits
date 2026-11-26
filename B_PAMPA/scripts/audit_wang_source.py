"""Read the PAMPA logarithm and SEM directly from Wang et al., Table S1."""
import argparse
import csv
import hashlib
from pathlib import Path
import re
import subprocess
from tempfile import TemporaryDirectory
from urllib.request import Request, urlopen

ROOT=Path(__file__).resolve().parents[1]
URL='https://ars.els-cdn.com/content/image/1-s2.0-S022352341530012X-mmc1.pdf'
LIMITS={29:'poor_solubility',30:'poor_solubility',31:'poor_solubility',46:'not_determined',49:'not_determined',
        50:'upper_bound',59:'zero_reported_no_numeric_log',60:'zero_reported_no_numeric_log'}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pdf',type=Path)
    p.add_argument('--output',type=Path,default=ROOT/'configs/wang2015_pampa.csv')
    a=p.parse_args()
    with TemporaryDirectory() as temp:
        f=a.pdf
        if f is None:
            f=Path(temp)/'SI.pdf'
            with urlopen(Request(URL,headers={'User-Agent':'Mozilla/5.0'}),timeout=45) as r:f.write_bytes(r.read())
        expected='5a014fdf2145ce5cf5bca3397cf47ce4323044707f819b0e98a0d6518228e6bd'
        if hashlib.sha256(f.read_bytes()).hexdigest()!=expected:raise ValueError('Source PDF checksum mismatch')
        t=subprocess.check_output(['pdftotext','-f','14','-l','15','-layout',str(f),'-'],text=True).split('Controls')[0]
    rows=[]
    for line in t.splitlines():
        start=re.match(r'\s*(\d+)\s+',line)
        if not start:continue
        i=int(start[1])
        v=re.search(r'(-?\d+\.\d+)\s*±\s*(\d+\.\d+)\s*$',line)
        if (v is None)!=(i in LIMITS):raise ValueError(f'Unexpected censored compound {i}')
        rows.append(dict(record_id=f'CycPeptMPDB_record:{1044+i}',source_compound=f'Compound.{i}',
                         pampa_log10_cm_s=f'{float(v[1])-6:.2f}' if v else '',reported_sem_log10=v[2] if v else '',
                         observation=LIMITS.get(i,'quantified'),log10_upper_bound=-8 if i==50 else '',mean_recovery_lt50pct_reported=int(i in [12,16])))
    assert [r['source_compound'] for r in rows]==[f'Compound.{i}' for i in range(1,63)]
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader();w.writerows(rows)
if __name__=='__main__':main()
