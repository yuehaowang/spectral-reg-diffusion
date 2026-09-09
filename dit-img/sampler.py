import torch


def euler_sampler(
    model, x0, y=None, null_cond=None,
    num_steps=100, cfg=2.0, sampler_min_t=0, sampler_max_t=1.0):

    xt = x0

    ts = torch.linspace(sampler_min_t, sampler_max_t, num_steps + 1).to(x0.device)
    dt = (sampler_max_t - sampler_min_t) / num_steps

    for t in ts:
        t = t.expand(x0.shape[0])
    
        vt = model(xt, t, y)
        if null_cond is not None:
            vu = model(xt, t, null_cond)
            vt = vu + cfg * (vt - vu)
        xt = xt + dt * vt
    
    return xt
