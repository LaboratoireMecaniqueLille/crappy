import crappy

dis = crappy.blocks.DISCorrel('Webcam',
    fields=['x','y'],
    labels=['t(s)','x(pix)','y(pix)'])

graph = crappy.blocks.Grapher(('x(pix)','y(pix)'))
crappy.link(dis,graph)
crappy.start()
