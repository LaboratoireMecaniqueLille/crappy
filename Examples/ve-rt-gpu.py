import crappy


#h,w = 6004,7920
h,w = 1024,1024
ps = 500 # patch size (x and y)

patches = [
    (0,w//2-ps//2,ps,ps), # Top
    (h//2-ps,w-ps,ps,ps), # Right
    (h-ps,w//2-ps-2,ps,ps), # Bottom
    (h//2-ps,0,ps,ps)] # Left

cam_kw = dict(
    exposure=8000, # µs
    height=h,
    width=w)

ve = crappy.blocks.GPUVE('Xiapi',patches,verbose=True,cam_kwargs=cam_kw)
graphy = crappy.blocks.Grapher(('t(s)','p0y'),('t(s)','p2y'))
graphx = crappy.blocks.Grapher(('t(s)','p1x'),('t(s)','p3x'))

crappy.link(ve,graphx)
crappy.link(ve,graphy)


def compute_strain(d):
  d['Exx(%)'] = (d['p3x']-d['p1x'])/(w-ps)*100
  d['Eyy(%)'] = (d['p0y']-d['p2y'])/(h-ps)*100
  return d


graphstrain = crappy.blocks.Grapher(('t(s)','Exx(%)'),('t(s)','Eyy(%)'))
crappy.link(ve,graphstrain,modifier=compute_strain)
crappy.start()
