import crappy

mic = crappy.blocks.Microphone(save_file='test.wav',channels=1)

crappy.start(high_prio=True)
