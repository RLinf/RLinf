import torch
from torch import nn
class TouchEncoder(nn.Module):
 def __init__(self):
  super().__init__();layers=[]
  for a,b in [(6,32),(32,64),(64,128)]:layers.extend([nn.Conv2d(a,b,3,2,1),nn.GroupNorm(8,b),nn.GELU()])
  self.cnn=nn.Sequential(*layers,nn.AdaptiveAvgPool2d(1),nn.Flatten(),nn.Linear(128,64))
  self.meta=nn.Sequential(nn.Linear(67,64),nn.GELU(),nn.Linear(64,64))
 def forward(self,x,meta):
  shape=x.shape[:3];flat=x.reshape(-1,6,140,80);v=meta[...,0].reshape(-1)>0
  z=flat.new_zeros((len(flat),64))
  if v.any():z[v]=self.cnn(flat[v])
  z=self.meta(torch.cat([z,meta.reshape(-1,3)],-1));z=z*v[:,None]
  return z.reshape(*shape,64)
class ProgressNet(nn.Module):
 def __init__(self,variant='baseline'):
  super().__init__();assert variant in ['baseline','touch_add','touch_concat'];self.variant=variant;self.mode='task4';dim=960 if variant=='touch_concat' else 832
  self.rlt_mlp=nn.Sequential(nn.LayerNorm(2048),nn.Linear(2048,512),nn.GELU(),nn.Linear(512,768))
  self.state_mlp=nn.Sequential(nn.Linear(16,64),nn.GELU(),nn.Linear(64,64))
  self.positions=nn.Parameter(torch.zeros(1,5,dim));layer=nn.TransformerEncoderLayer(dim,4,dim*4,.1,activation='gelu',batch_first=True,norm_first=True)
  self.temporal=nn.TransformerEncoder(layer,4,enable_nested_tensor=False)
  for layer in self.temporal.layers:
   for p in layer.parameters():
    if p.ndim>1:nn.init.xavier_uniform_(p)
  self.heads=nn.ModuleList([nn.Sequential(nn.Linear(dim,128),nn.GELU(),nn.Linear(128,1)) for _ in range(4)])
  self.register_buffer('state_mean',torch.zeros(16));self.register_buffer('state_std',torch.ones(16))
  if variant!='baseline':self.touch=TouchEncoder()
  if variant=='touch_add':
   self.touch_projection=nn.Sequential(nn.Linear(134,256),nn.GELU(),nn.Linear(256,832));nn.init.zeros_(self.touch_projection[-1].weight);nn.init.zeros_(self.touch_projection[-1].bias)
 def forward(self,rlt,state,valid,task,touch=None,meta=None):
  assert rlt.shape[1:]==(5,2048) and state.shape[1:]==(5,16) and valid[:,-1].all()
  assert ((task>=0)&(task<4)).all()
  z=torch.cat([self.rlt_mlp(rlt),self.state_mlp((state-self.state_mean)/self.state_std)],-1)
  if self.variant!='baseline':
   assert touch.shape[1:]==(5,2,6,140,80) and meta.shape[1:]==(5,2,3)
   t=self.touch(touch,meta).flatten(-2)
   if self.variant=='touch_concat':z=torch.cat([z,t],-1)
   else:z=z+self.touch_projection(torch.cat([t,meta.flatten(-2)],-1))*(meta[...,0].any(-1))[...,None]
  z=z.masked_fill(~valid[...,None],0)+self.positions;h=self.temporal(z,src_key_padding_mask=~valid)[:,-1];out=h.new_zeros(len(h))
  for i,head in enumerate(self.heads):
   use=task==i
   if use.any():out[use]=4*torch.sigmoid(head(h[use]).squeeze(-1))
  return out
