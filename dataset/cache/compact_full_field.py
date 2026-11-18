"""Subsample an existing full-field cache and audit direct float16 conversion.

Outputs stay float32: unsafe casts are measured, never silently clipped/saved.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import json
from pathlib import Path
import shutil
import time

import numpy as np

from dataset.cache.full_field import ROOT, atomic_json, atomic_npy, select_ids, source_signature


def case_path(root, record):
    if record['domain'] == 'train':
        return Path(root)/'train'/record['family']/record['case']
    return Path(root)/'rocket'/record['case']


def audit_half(array, columns):
    with np.errstate(over='ignore', under='ignore', invalid='ignore'):
        half = array.astype(np.float16).astype(np.float32)
    result = {}
    for j, name in enumerate(columns):
        x = array[:,j].astype(np.float64)
        y = half[:,j].astype(np.float64)
        valid = np.isfinite(y)
        dx = np.abs(y[valid] - x[valid])
        nonzero = x[valid] != 0
        relative = dx[nonzero] / np.abs(x[valid][nonzero])
        result[name] = dict(count=len(x), nonzero_count=int(np.count_nonzero(x)),
            overflow_count=int(np.isinf(y).sum()), nan_count=int(np.isnan(y).sum()),
            underflow_to_zero_count=int(((x!=0)&(y==0)).sum()),
            finite_cast_count=int(valid.sum()), raw_min=float(x.min()), raw_max=float(x.max()),
            max_abs_error_finite=float(dx.max(initial=0)), sum_abs_error_finite=float(dx.sum()),
            sum_sq_error_finite=float(np.dot(dx,dx)), sum_sq_reference_finite=float(np.dot(x[valid],x[valid])),
            max_relative_error_finite=float(relative.max(initial=0)),
            relative_error_gt_1pct_count=int((relative>.01).sum()))
    return result


def compact_case(task):
    old, source_root, output_root, policy = task
    source = case_path(source_root,old)
    dest = case_path(output_root,old)
    signature = source_signature([source/f'{s}{suffix}.npy' for s in ['surface','volume']
                                  for suffix in ['', '_indices']])
    record = dict(old, schema_version=2, dtype='float32', index_dtype='int32',
                  parent_cache=str(source), parent_cache_signature=signature, **policy)
    audit = {}
    bytes_written = 0
    for side in ['surface','volume']:
        rows = np.load(source/f'{side}.npy',mmap_mode='r')
        original_ids = np.load(source/f'{side}_indices.npy',mmap_mode='r')
        take = select_ids(len(rows),policy[f'{side}_cap'],old['case']+'/'+side+'/compact',policy['seed'])
        data = np.array(rows[take],dtype=np.float32)
        ids64 = np.asarray(original_ids[take],dtype=np.int64)
        if not np.isfinite(data).all() or len(data)!=min(old[f'{side}_available_points'],policy[f'{side}_cap']):
            raise ValueError(f'{source}/{side}: invalid input or unsupported enlargement')
        if np.any(np.diff(ids64)<=0) or ids64.min()<0 or ids64.max()>=old[f'{side}_source_points']:
            raise ValueError(f'{source}/{side}: invalid source row IDs')
        if ids64.max()>np.iinfo(np.int32).max:
            raise ValueError('Source row indices exceed int32 range')
        ids = ids64.astype(np.int32)
        record[f'{side}_points'] = len(data)
        record[f'{side}_parent_points'] = len(rows)
        audit[side] = audit_half(data,record[f'{side}_columns'])
        atomic_npy(dest/f'{side}.npy',data)
        atomic_npy(dest/f'{side}_indices.npy',ids)
        # Readback checks the actual stored rows and composed original indices.
        saved = np.load(dest/f'{side}.npy',mmap_mode='r')
        saved_ids = np.load(dest/f'{side}_indices.npy',mmap_mode='r')
        np.testing.assert_array_equal(saved,data)
        np.testing.assert_array_equal(saved_ids,ids64)
        assert saved.dtype==np.float32 and saved_ids.dtype==np.int32
        for name in [f'{side}.npy',f'{side}_indices.npy']:
            bytes_written += (dest/name).stat().st_size
    for name in ['source_metadata.json','source_field_data.json']:
        if (source/name).exists():
            dest.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(source/name,dest/name)
    atomic_json(dest/'record.json',record)
    return record, audit, bytes_written


def merge_metric(total, metric, case):
    total['cases'] = total.get('cases',0)+1
    for kind in ['overflow','nan','underflow_to_zero']:
        total[f'{kind}_cases'] = total.get(f'{kind}_cases',0)+int(metric[f'{kind}_count']>0)
    for key,value in metric.items():
        if key=='raw_min':
            total[key]=min(total.get(key,value),value)
        elif key=='raw_max' or key.startswith('max_'):
            if key not in total or value>total[key]:
                total[key]=value
                total[key+'_case']=case
        else:
            total[key]=total.get(key,0)+value


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=Path.home()/'zoujunhong/data/FullFieldCache/surface100000_volume300000')
    p.add_argument('--output',type=Path,default=Path.home()/'zoujunhong/data/FullFieldCache/surface32768_volume100000_fp32')
    p.add_argument('--workers',type=int,default=12)
    p.add_argument('--surface-cap',type=int,default=32768)
    p.add_argument('--volume-cap',type=int,default=100000)
    p.add_argument('--seed',type=int,default=20261026)
    p.add_argument('--limit',type=int)
    args=p.parse_args()
    if args.source.resolve()==args.output.resolve():p.error('Output must differ from input')
    if min(args.surface_cap,args.volume_cap,args.workers)<=0:p.error('Caps/workers must be positive')
    old_index=json.loads((args.source/'index.json').read_text())
    if not old_index['ready']:raise ValueError('Source cache not ready')
    if args.surface_cap>old_index['surface_cap'] or args.volume_cap>old_index['volume_cap']:
        raise ValueError('Cannot enlarge a sampled cache')
    records=old_index['cases']
    if args.limit:
        records=[r for domain in ['train','rocket'] for r in [v for v in records if v['domain']==domain][:args.limit]]
    policy=dict(surface_cap=args.surface_cap,volume_cap=args.volume_cap,seed=args.seed)
    contract=dict(schema_version=2,dtype='float32',index_dtype='int32',**policy,
        sampling='uniform no-replacement subset of parent cache; original row IDs composed; no padding',
        parent_cache=str(args.source.resolve()),parent_index_signature=source_signature([args.source/'index.json']))
    if (args.output/'contract.json').exists() and json.loads((args.output/'contract.json').read_text())!=contract:
        raise ValueError('Existing output contract differs')
    atomic_json(args.output/'contract.json',contract)
    ready,errors,fields,per_case=[],[],{},[]
    total_bytes=0;start=time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        pending={pool.submit(compact_case,(r,str(args.source),str(args.output),policy)):r for r in records}
        for count,future in enumerate(as_completed(pending),1):
            r=pending[future]
            try:
                record,audit,size=future.result();ready.append(record);total_bytes+=size
                counts=dict(domain=r['domain'],case=r['case'])
                for side,columns in audit.items():
                    for col,m in columns.items():
                        key=f"{r['domain']}/{side}/{col}"
                        merge_metric(fields.setdefault(key,{}),m,r['case'])
                    for kind in ['overflow','nan','underflow_to_zero']:
                        counts[f'{side}_{kind}']=sum(m[f'{kind}_count'] for m in columns.values())
                per_case.append(counts)
            except Exception as exc:
                errors.append(dict(domain=r['domain'],case=r['case'],error=repr(exc)))
                print('ERROR',errors[-1],flush=True)
            if count%100==0 or count==len(records):
                progress=dict(done=count,total=len(records),successful=len(ready),errors=errors,
                              seconds=round(time.time()-start,2))
                atomic_json(args.output/'progress.json',progress)
                print(json.dumps(progress),flush=True)
    ready.sort(key=lambda r:(r['domain'],r['case']))
    for metric in fields.values():
        n=max(metric['finite_cast_count'],1)
        metric['mae_finite']=metric['sum_abs_error_finite']/n
        metric['rmse_finite']=(metric['sum_sq_error_finite']/n)**.5
        denom=metric['sum_sq_reference_finite']
        metric['relative_l2_finite']=(metric['sum_sq_error_finite']/denom)**.5 if denom else 0.
    audit_summary=dict(cases=len(ready),conversion='direct float32 -> float16 -> float32, without scaling/clipping',
        error_metrics_scope='finite cast elements only; overflow/nan/underflow counts reported separately',
        direct_float16_all_finite=not any(m['overflow_count'] or m['nan_count'] for m in fields.values()),fields=fields)
    atomic_json(args.output/'float16_audit.json',audit_summary)
    with (args.output/'float16_cases.csv').open('w',newline='') as f:
        if per_case:
            writer=csv.DictWriter(f,fieldnames=list(per_case[0]));writer.writeheader();writer.writerows(per_case)
    complete=not errors and args.limit is None and len(ready)==len(old_index['cases'])
    index=dict(**contract,ready=complete,cases=ready,excluded=old_index['excluded'],errors=errors)
    atomic_json(args.output/('index.json' if complete else 'partial_index.json'),index)
    atomic_json(args.output/'excluded.json',old_index['excluded'])
    validation=dict(cases=len(ready),ready=complete,array_and_index_bytes=total_bytes,
        all_stored_arrays_float32_finite=not errors,all_stored_rows_equal_parent_selection=not errors,
        all_indices_int32_exact_and_unique=not errors,errors=errors)
    atomic_json(args.output/'validation.json',validation)
    with (args.output/'cases.csv').open('w',newline='') as f:
        keys=['domain','case','split','surface_points','volume_points']
        writer=csv.DictWriter(f,fieldnames=keys,extrasaction='ignore');writer.writeheader();writer.writerows(ready)
    print('COMPLETE',json.dumps(validation),flush=True)
    if errors:raise SystemExit(1)


if __name__=='__main__':main()
