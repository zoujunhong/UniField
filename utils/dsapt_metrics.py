import torch
FIELDS=('surface_p','volume_p','volume_u')
STAT_KEYS={'surface_p':'surface_pressure','volume_p':'volume_pressure','volume_u':'volume_velocity'}
def relative_l2(p,y):
 return (p-y).flatten(1).norm(dim=1)/y.flatten(1).norm(dim=1).clamp_min(1e-12)
def validation_metrics(pred,b,c):
 result={}
 fields=[field for field in FIELDS if field in pred and 'target_'+field in b]
 for f in fields:
  p=pred[f].float();y=b['target_'+f].float();key=STAT_KEYS[f];stats=c['label_stats']
  mean=y.new_tensor(stats[key+'_mean']);std=y.new_tensor(stats[key+'_std'])
  result[f+'_rel_l2_standard']=float(relative_l2(p,y).mean())
  result[f+'_rel_l2_raw']=float(relative_l2(p*std+mean,y*std+mean).mean())
  result[f+'_mse_standard']=float((p-y).square().mean())
 result['joint_rel_l2_standard']=sum(result[f+'_rel_l2_standard'] for f in fields)/len(fields)
 return result
