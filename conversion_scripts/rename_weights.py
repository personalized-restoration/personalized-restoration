import torch
import os

mdl_og = '' ##Path to the original Stable Diffusion 2.1 checkpoint you downloaded
mdl_our = '' ## Path to the ckpt of the DreamBooth model you created in the previous step

transform_dict = {'first_stage_model.encoder.mid.attn_1.q.weight':'first_stage_model.encoder.mid.attn_1.to_q.weight',
                    'first_stage_model.encoder.mid.attn_1.q.bias':'first_stage_model.encoder.mid.attn_1.to_q.bias',
                    'first_stage_model.encoder.mid.attn_1.k.weight':'first_stage_model.encoder.mid.attn_1.to_k.weight',
                    'first_stage_model.encoder.mid.attn_1.k.bias':'first_stage_model.encoder.mid.attn_1.to_k.bias',
                    'first_stage_model.encoder.mid.attn_1.v.weight':'first_stage_model.encoder.mid.attn_1.to_v.weight',
                    'first_stage_model.encoder.mid.attn_1.v.bias':'first_stage_model.encoder.mid.attn_1.to_v.bias',
                    'first_stage_model.encoder.mid.attn_1.proj_out.weight':'first_stage_model.encoder.mid.attn_1.to_out.0.weight',
                    'first_stage_model.encoder.mid.attn_1.proj_out.bias':'first_stage_model.encoder.mid.attn_1.to_out.0.bias',
                    'first_stage_model.decoder.mid.attn_1.q.weight':'first_stage_model.decoder.mid.attn_1.to_q.weight',
                    'first_stage_model.decoder.mid.attn_1.q.bias':'first_stage_model.decoder.mid.attn_1.to_q.bias',
                    'first_stage_model.decoder.mid.attn_1.k.weight':'first_stage_model.decoder.mid.attn_1.to_k.weight',
                    'first_stage_model.decoder.mid.attn_1.k.bias':'first_stage_model.decoder.mid.attn_1.to_k.bias',
                    'first_stage_model.decoder.mid.attn_1.v.weight':'first_stage_model.decoder.mid.attn_1.to_v.weight',
                    'first_stage_model.decoder.mid.attn_1.v.bias':'first_stage_model.decoder.mid.attn_1.to_v.bias',
                    'first_stage_model.decoder.mid.attn_1.proj_out.weight':'first_stage_model.decoder.mid.attn_1.to_out.0.weight',
                    'first_stage_model.decoder.mid.attn_1.proj_out.bias':'first_stage_model.decoder.mid.attn_1.to_out.0.bias'}





print('Loading og model')
m1 = torch.load(mdl_og)
m1_params = m1['state_dict']
m1 = m1_params.keys()

print('Loading our model')
m2 = torch.load(mdl_our)
m2_params = m2['state_dict']
m2 = m2_params.keys()


for kk in m1:
    if kk in m2:
        
        m1_params[kk] = m2_params[kk].clone()
        
        
    
    elif kk in transform_dict.keys():
        bb = m2_params[transform_dict[kk]].clone()
        if len(bb.shape)==2:
            bb = bb.unsqueeze(-1).unsqueeze(-1)
        
        m1_params[kk] = bb
        
    else:
        print(kk)
        

out_dict = {}
out_dict['state_dict'] = m1_params

save_path = '' ## Add path to the location where you wish the save the corrected, renamed model
torch.save(out_dict,save_path)
