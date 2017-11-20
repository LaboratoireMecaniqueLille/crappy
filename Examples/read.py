
import crappy

if __name__ == "__main__":
  l = list(crappy.inout.in_list.keys())
  for i,c in enumerate(l):
    print(i,c)
  name = l[int(input("What board do you want to use ?> "))]

  m = crappy.blocks.IOBlock(name,labels=['t(s)','chan0'],verbose=True)

  g = crappy.blocks.Grapher(('t(s)','chan0'))

  crappy.link(m,g)

  crappy.start()
