import crappy

labjack = crappy.technical.LabJack(sensor={'mode': 'streamer', 'scan_rate_per_channel': 10000, 'scans_per_read': 1000})
streamer = crappy.blocks.Streamer(sensor=labjack, mean=100)
grapher = crappy.blocks.Grapher(('time(sec)', 'AIN0'), length=10)

crappy.link(streamer, grapher)

crappy.start()
