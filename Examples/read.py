
import crappy


for i,c in enumerate(crappy.inout.in_list):
  print(i,c)
name = list(crappy.inout.in_list.keys())[int(input(
                    "What board do you want to use ?> "))]

m = crappy.blocks.IOBlock(name,labels=['t(s)','chan0'],verbose=True)

g = crappy.blocks.Grapher(('t(s)','chan0'))

crappy.link(m,g)

crappy.start()
